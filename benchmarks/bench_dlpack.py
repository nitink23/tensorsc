"""DLPack versus existing buffer conversion overhead (release builds only)."""
import argparse
import json
import platform
import statistics
import timeit
import numpy as np
import tensorsc as ts


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--save',required=True)
    args=parser.parse_args()
    if ts._debug_build: raise SystemExit('Build with maturin develop --release first')
    try: import torch
    except ImportError: torch=None
    results=[]
    for n in [16,262144]:
        a=np.arange(n,dtype=np.float32)
        x=ts.tensor(a)
        cases={
            'from_numpy_buffer':lambda:ts.from_numpy(a),
            'from_numpy_dlpack':lambda:ts.from_dlpack(a),
            'to_numpy_buffer':lambda:np.asarray(x),
            'to_numpy_dlpack':lambda:np.from_dlpack(x),
        }
        if torch is not None:
            t=torch.from_numpy(a)
            cases['from_torch_dlpack']=lambda:ts.from_dlpack(t)
            cases['to_torch_dlpack']=lambda:torch.from_dlpack(x)
        for name,fn in cases.items():
            result=fn()
            if isinstance(result,ts.Tensor): check=result.numpy()
            elif torch is not None and isinstance(result,torch.Tensor): check=result.numpy()
            else: check=result
            np.testing.assert_array_equal(check,a)
            assert np.shares_memory(check,a if name.startswith('from_') else np.asarray(x))
            seconds=statistics.median(timeit.repeat(fn,number=10000,repeat=5))/10000
            results.append(dict(operation=name,elements=n,seconds=seconds))
            print(f'{name:22} n={n:6} {seconds*1e6:.3f} us',flush=True)
    with open(args.save,'w') as f:
        json.dump(dict(build='release',python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,torch=getattr(torch,'__version__',None),method='median of 5 batches of 10000 calls, inputs and pointer/value checks outside timing',results=results),f,indent=2)

if __name__=='__main__':main()
