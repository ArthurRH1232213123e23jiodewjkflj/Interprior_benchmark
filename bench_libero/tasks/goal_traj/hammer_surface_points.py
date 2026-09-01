"""Sample N surface points on the REAL DexJoCo hammer (MuJoCo primitive geoms).
Hammer local frame, meters. Geoms from xmls/hammer.xml.
Output: points_obj [N,3] in hammer body local frame.
"""
import numpy as np

def _quat_wxyz_to_R(q):
    w,x,y,z = q
    n = np.sqrt(w*w+x*x+y*y+z*z); w,x,y,z = w/n,x/n,y/n,z/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ])

def _box_surface(size, n, rng):
    # size = half-extents; sample n points on the 6 faces, area-weighted
    sx,sy,sz = size
    areas = np.array([sy*sz,sy*sz, sx*sz,sx*sz, sx*sy,sx*sy])
    faces = rng.choice(6, size=n, p=areas/areas.sum())
    u = rng.uniform(-1,1,size=(n,2))
    pts = np.zeros((n,3))
    for i,f in enumerate(faces):
        a,b = u[i]
        if f==0: pts[i]=[ sx, a*sy, b*sz]
        elif f==1: pts[i]=[-sx, a*sy, b*sz]
        elif f==2: pts[i]=[a*sx,  sy, b*sz]
        elif f==3: pts[i]=[a*sx, -sy, b*sz]
        elif f==4: pts[i]=[a*sx, b*sy,  sz]
        else:      pts[i]=[a*sx, b*sy, -sz]
    return pts

def _cyl_surface(radius, halflen, n, rng):
    # axis along local z; lateral + 2 caps, area-weighted
    lat = 2*np.pi*radius*(2*halflen); cap = np.pi*radius*radius
    areas = np.array([lat, cap, cap]); which = rng.choice(3,size=n,p=areas/areas.sum())
    pts = np.zeros((n,3)); th = rng.uniform(0,2*np.pi,size=n)
    for i,wsel in enumerate(which):
        if wsel==0:
            z = rng.uniform(-halflen,halflen)
            pts[i]=[radius*np.cos(th[i]), radius*np.sin(th[i]), z]
        else:
            r = radius*np.sqrt(rng.uniform(0,1)); zc = halflen if wsel==1 else -halflen
            pts[i]=[r*np.cos(th[i]), r*np.sin(th[i]), zc]
    return pts

# geoms: (kind, pos, quat_wxyz, size)
GEOMS = [
    ("box", [0,0,0],          [1,0,0,0],                 [0.0254,0.0254,0.1271]),  # handle
    ("box", [0,0,0.1575],     [1,0,0,0],                 [0.061,0.0305,0.0305]),   # head
    ("cyl", [0.0671,0,0.1575],[0.707106,0,0.707106,0],   [0.0244,0.0061]),         # neck
    ("cyl", [0.0854,0,0.1575],[0.707106,0,0.707106,0],   [0.0305,0.0122]),         # face
    ("box", [-0.061,0,0.1575],[0.9238795,0,0.3826834,0], [0.0215,0.029,0.0215]),   # claw
]

def dense_surface(n_per=4000, seed=0):
    rng = np.random.default_rng(seed); allp=[]
    for kind,pos,quat,size in GEOMS:
        R=_quat_wxyz_to_R(np.array(quat,float)); pos=np.array(pos,float)
        loc = _box_surface(size,n_per,rng) if kind=="box" else _cyl_surface(size[0],size[1],n_per,rng)
        allp.append(loc@R.T + pos)
    return np.concatenate(allp,0)

def fps(points, N, seed=0):
    rng=np.random.default_rng(seed); M=len(points)
    idx=np.zeros(N,int); idx[0]=rng.integers(M)
    d=np.full(M,np.inf)
    for i in range(1,N):
        d=np.minimum(d, np.sum((points-points[idx[i-1]])**2,1))
        idx[i]=int(np.argmax(d))
    return points[idx]

def sample_points_obj(N=16, seed=0):
    return fps(dense_surface(seed=seed), N, seed=seed).astype(np.float32)

if __name__=="__main__":
    p=sample_points_obj(16)
    print("points_obj shape:", p.shape)
    print("x-range %.3f..%.3f  z-range %.3f..%.3f"%(p[:,0].min(),p[:,0].max(),p[:,2].min(),p[:,2].max()))
    # sanity: claw side (x<0) and face side (x>0.08) both represented?
    print("has claw(-x):", (p[:,0]<-0.03).any(), " has face(+x):", (p[:,0]>0.07).any(), " has handle-bottom:", (p[:,2]<-0.1).any())
