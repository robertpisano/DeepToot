import numpy as np

def geodesic_distance(x,y):
    dot_product = np.sum(x*y)
    mag_x = np.linalg.norm(x)
    mag_y = np.linalg.norm(y)
    cosine = dot_product/(mag_x*mag_y)
    if cosine>1: cosine = 1
    if cosine<-1: cosine = -1
    return np.arccos(cosine)

def log_map(x,y):
    d = geodesic_distance(x,y)
    temp = y - np.sum(x*y) * x
    if np.linalg.norm(temp) != 0:
        mapped_value = d * (temp/np.linalg.norm(temp))
    else:
        mapped_value = np.array([0.0,0.0,0.0])
    return mapped_value

def exp_map(p,v):
    mag_v = np.linalg.norm(v)
    if mag_v == 0:
        return p
    v_normalized = v/mag_v
    mapped_value = p * np.cos(mag_v) + v_normalized * np.sin(mag_v)
    return mapped_value

def parallel_transport(v,p,q):
    logmap1 = log_map(p,q)
    logmap2 = log_map(q,p)
    if np.linalg.norm(logmap1)!=0 and np.linalg.norm(logmap2)!=0:
        transported_value = v - (np.dot(logmap1 , v)/geodesic_distance(p,q)) * (logmap1+logmap2)
    else:
        transported_value = v
    return transported_value
