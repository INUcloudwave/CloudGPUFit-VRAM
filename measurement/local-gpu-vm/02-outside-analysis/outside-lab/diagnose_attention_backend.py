#!/usr/bin/env python3
"""Separate profiler process; its memory is never an outside measurement."""
import argparse,hashlib,inspect,json,os,traceback
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--attention',choices=['eager','sdpa'],required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=False)
 meta=dict(status='starting',purpose='diagnostic_only_not_outside_repeat',attention=a.attention,pid=os.getpid(),module_loading=os.environ.get('CUDA_MODULE_LOADING'))
 try:
  import torch as t
  import transformers as tr
  from transformers.models.qwen2 import modeling_qwen2
  from transformers.integrations import sdpa_attention
  t.cuda.init();assert t.cuda.device_count()==1;t.set_num_threads(4);t.manual_seed(20261006)
  t.backends.cuda.matmul.allow_tf32=False;t.backends.cudnn.allow_tf32=False;t.backends.cudnn.benchmark=False
  allocation=t.empty(1,device='cuda');del allocation
  model=tr.AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B',revision='272aa6d30271028cdf113345b1e82a660437edaa',torch_dtype=t.bfloat16,low_cpu_mem_usage=True,device_map={'':'cuda:0'},use_safetensors=True,local_files_only=True,attn_implementation=a.attention).to(device='cuda',dtype=t.bfloat16).eval()
  cfg=model.config;cfg.output_attentions=False;cfg.output_hidden_states=False
  inputs={'input_ids':t.randint(3,cfg.vocab_size,(1,256)),'attention_mask':t.ones(1,256,dtype=t.long)}
  for s in [cfg.pad_token_id,cfg.eos_token_id,cfg.bos_token_id]:
   if isinstance(s,int):inputs['input_ids'][inputs['input_ids']==s]=3
  meta.update(torch=t.__version__,transformers=tr.__version__,cuda_build=t.version.cuda,selected_transformers_attention=cfg._attn_implementation,gpu_uuid=str(t.cuda.get_device_properties(0).uuid),input_sha256={k:hashlib.sha256(v.contiguous().view(t.uint8).numpy().tobytes()).hexdigest() for k,v in inputs.items()})
  for name,obj in [('qwen2_attention_forward',modeling_qwen2.Qwen2Attention.forward),('sdpa_attention_forward',sdpa_attention.sdpa_attention_forward)]:
   src=inspect.getsource(obj);(out/(name+'.py.txt')).write_text(src);meta[name+'_source_sha256']=hashlib.sha256(src.encode()).hexdigest()
  results=[]
  for stage in ['first_forward','generate64']:
   with t.profiler.profile(activities=[t.profiler.ProfilerActivity.CPU],record_shapes=False,profile_memory=False,with_stack=False) as prof:
    with t.inference_mode():
     gi={k:v.to('cuda') for k,v in inputs.items()}
     if stage=='first_forward':
      r=model(**gi);r.logits.cpu();assert bool(t.isfinite(r.logits).all().item());del r
     else:
      r=model.generate(**gi,do_sample=False,num_beams=1,use_cache=True,min_new_tokens=64,max_new_tokens=64,pad_token_id=cfg.pad_token_id if cfg.pad_token_id is not None else cfg.eos_token_id,return_dict_in_generate=False,output_scores=False,output_attentions=False,output_hidden_states=False);r.cpu();assert r.shape[1]==320;del r
     t.cuda.synchronize()
   ops=[dict(operator=e.key,count=e.count) for e in prof.key_averages()]
   results.append(dict(stage=stage,attention_operators=[x for x in ops if any(k in x['operator'].lower() for k in ['attention','flash','efficient','softmax'])],all_operator_counts=ops))
   del prof
  meta.update(status='ok',profiles=results)
 except BaseException:meta.update(status='error',traceback=traceback.format_exc());raise
 finally:(out/'diagnostic.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__':main()
