import json, timeit, statistics, sys
import numpy as np
import tensorsc as ts
rows=[]
for m,k,n in [(128,128,128),(256,256,256),(512,512,512),(37,65,91),(128,1,128),(1,512,512)]:
    rng=np.random.default_rng(0)
    a=rng.normal(size=(m,k)).astype('float32'); b=rng.normal(size=(k,n)).astype('float32')
    x,y=ts.tensor(a),ts.tensor(b)
    np.testing.assert_allclose((x@y).numpy(),a@b,rtol=1e-4,atol=1e-4)
    t=statistics.median(timeit.repeat(lambda:x@y,number=5,repeat=5))/5
    rows.append(dict(shape=[m,k,n],seconds=t))
    print((m,k,n), t*1000, 'ms',flush=True)
with open(sys.argv[1],'w') as f: json.dump(rows,f,indent=2)
