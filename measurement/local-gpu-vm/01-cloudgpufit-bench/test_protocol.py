"""CPU checks for isolation/protocol logic; these do not replace GPU validation."""
import argparse
import ast
import copy
import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

runner=load('run_bench_v3')
comparator=load('compare_results')

def worker_helpers():
    tree=ast.parse((ROOT/'bench_worker_v3.py').read_text(encoding='utf-8'))
    names={'load_or_reuse','run_shared','wait_mode_idle','cleanup_between_modes'}
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    env={'copy':copy,'Path':Path,'json':json,'MODE_ORDER':runner.MODES}
    exec(compile(ast.Module(body=selected,type_ignores=[]),'<worker helpers>','exec'),env)
    return env

class State(list):
    def clone(self):return State(self)

class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.plan=json.loads((ROOT/'benchmark_plan_v2.json').read_text(encoding='utf-8'))
        self.profile=self.plan['execution_profiles']['local_core']

    def test_all_conditions_retained(self):
        cases=runner.select_cases(self.plan,self.profile)
        self.assertEqual(len(cases),342)
        self.assertEqual(sum(1 for c in cases for m in runner.MODES if c['workload_type']!='vision_cls' or m not in ('first_logits','tokens')),1548)

    def test_pilot_covers_every_model_and_dtype_expansion(self):
        nine=runner.select_cases(self.plan,self.profile,case_set='baseline9')
        self.assertEqual(len(nine),9)
        self.assertEqual(len({c['model'] for c in nine}),9)
        self.assertEqual({c['dtype'] for c in nine},{'fp16'})
        self.assertEqual(len(runner.select_cases(self.plan,self.profile,case_set='baseline27')),27)

    def test_measurement_body_identical_to_original(self):
        def get(path,name):
            tree=ast.parse(path.read_text(encoding='utf-8'))
            return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
        for name in ['invoke','cache_bytes','storage_bytes']:
            self.assertEqual(ast.dump(get(ROOT/'bench_worker_legacy.py',name)),ast.dump(get(ROOT/'bench_worker_v3.py',name)))

    def test_reuse_restores_rng_and_config_but_never_crosses_case(self):
        env=worker_helpers();state={'cpu':State([1]),'cuda':State([2])};calls=[]
        cuda=types.SimpleNamespace(get_rng_state=lambda:state['cuda'],set_rng_state=lambda s:state.update(cuda=s.clone()))
        env['torch']=types.SimpleNamespace(cuda=cuda,get_rng_state=lambda:state['cpu'],set_rng_state=lambda s:state.update(cpu=s.clone()))
        def factory():
            calls.append(1);state.update(cpu=State([11]),cuda=State([22]))
            return types.SimpleNamespace(config={'a':1},generation_config={'b':2})
        session={};key=('case1','model','fp16','revision')
        model,reused=env['load_or_reuse'](session,key,factory)
        self.assertFalse(reused)
        model.config['a']=9;model.generation_config['b']=8;state.update(cpu=State([999]),cuda=State([888]))
        again,reused=env['load_or_reuse'](session,key,factory)
        self.assertTrue(reused);self.assertIs(again,model);self.assertEqual(len(calls),1)
        self.assertEqual(model.config,{'a':1});self.assertEqual(model.generation_config,{'b':2})
        self.assertEqual(state['cpu'],[11]);self.assertEqual(state['cuda'],[22])
        with self.assertRaises(RuntimeError):env['load_or_reuse'](session,('case2',*key[1:]),factory)

    def test_shared_order_and_one_factory_call(self):
        env=worker_helpers();calls=[];order=[]
        env['emit']=lambda *a,**k:None
        env['cleanup_between_modes']=lambda **kwargs:{}
        env['wait_mode_idle']=lambda *a:{}
        def main(args,session):
            if 'model' not in session:session['model']=object();calls.append(1)
            order.append((args.mode,args.iterations))
            Path(args.output).write_text(json.dumps({'case':{'case_id':'c'},'mode':args.mode,'repeat':0,'status':'ok'}))
        env['main']=main
        env['save']=lambda p,v:Path(p).write_text(json.dumps(v))
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            path=Path(d);(path/'plan.json').write_text(json.dumps({'cases':[{'case_id':'c','workload_type':'decoder_llm'}]}))
            args=argparse.Namespace(plan=str(path/'plan.json'),case='c',output=str(path/'{mode}.json'),
                                    modes=list(reversed(runner.MODES)),repeat=0,max_start_temp=36,idle_timeout=180,
                                    mode_iterations_json='{"memory":3,"tokens":3}',iterations=20,keep_allocator_cache=True)
            env['run_shared'](args)
        self.assertEqual(order,[(m,3 if m in ('memory','tokens') else 20) for m in runner.MODES])
        self.assertEqual(len(calls),1)

    def test_no_temperature_wait_accepts_warm_idle_gpu(self):
        env=worker_helpers();now=[0];sleeps=[]
        def clock():now[0]+=.01;return now[0]
        env['os']=types.SimpleNamespace(getpid=lambda:123)
        env['time']=types.SimpleNamespace(perf_counter=clock,sleep=lambda t:sleeps.append(t))
        env['pynvml']=types.SimpleNamespace(
            nvmlInit=lambda:None,nvmlDeviceGetHandleByIndex=lambda x:0,
            nvmlDeviceGetComputeRunningProcesses=lambda h:[types.SimpleNamespace(pid=123)],
            nvmlDeviceGetTemperature=lambda *a:60,NVML_TEMPERATURE_GPU=0,
            nvmlDeviceGetUtilizationRates=lambda h:types.SimpleNamespace(gpu=0),
            nvmlDeviceGetMemoryInfo=lambda h:types.SimpleNamespace(used=1024))
        state=env['wait_mode_idle'](0,180)
        self.assertEqual(state['temperature_c'],60)
        self.assertEqual(sleeps,[])

    def test_hybrid_isolates_first_logits_and_keeps_all_other_modes(self):
        groups=runner.mode_groups(runner.MODES,'hybrid')
        self.assertEqual(groups,[['performance','memory','tokens','components'],['first_logits']])
        self.assertEqual(sorted(m for g in groups for m in g),sorted(runner.MODES))
        self.assertTrue(runner.shared_worker(groups[0],'hybrid'))
        self.assertFalse(runner.shared_worker(groups[1],'hybrid'))
        cases=runner.select_cases(self.plan,self.profile,case_set='baseline9')
        self.assertEqual(runner.model_load_count(cases,runner.MODES,'hybrid',1),15)
        self.assertEqual(runner.model_load_count(cases,runner.MODES,'separate-mode',1),39)

    def test_cleanup_releases_garbage_without_emptying_allocator_in_hybrid(self):
        env=worker_helpers();calls=[]
        env['time']=types.SimpleNamespace(perf_counter=lambda:0)
        env['gc']=types.SimpleNamespace(collect=lambda:calls.append('gc'))
        env['torch']=types.SimpleNamespace(cuda=types.SimpleNamespace(
            synchronize=lambda:calls.append('sync'),empty_cache=lambda:calls.append('empty'),
            memory_allocated=lambda:100,memory_reserved=lambda:200))
        n=env['cleanup_between_modes'](keep_allocator_cache=True)
        self.assertNotIn('empty',calls);self.assertIn('gc',calls)
        self.assertTrue(n['allocator_cache_retained'])
        env['cleanup_between_modes'](keep_allocator_cache=False)
        self.assertIn('empty',calls)

    def test_comparison_flags_environment_and_memory_regression(self):
        data={'case':{'case_id':'c'},'mode':'memory','status':'ok','environment':{'gpu':'3090'},
              'raw_invocations':[{'elapsed_ms':1},{'elapsed_ms':1}],
              'inference_peak_allocated_bytes':100*2**20,'nvml_status':'ok'}
        rows,issues=comparator.compare_record(data,copy.deepcopy(data))
        self.assertTrue(all(r['screen']=='within_screen' for r in rows));self.assertEqual(issues,[])
        new=copy.deepcopy(data);new['environment']['gpu']='L4';new['inference_peak_allocated_bytes']*=2
        rows,issues=comparator.compare_record(data,new)
        self.assertIn('environment:gpu',issues)
        self.assertTrue(any(r['screen']=='investigate' for r in rows))

if __name__=='__main__':unittest.main(verbosity=2)
