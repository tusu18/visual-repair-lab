"""Evaluator-only vectorized specification, independent of repair-source code."""
import numpy as np


def detection_reference(state,settings):
    rows=state['detections'];w,h=state['frame']['width'],state['frame']['height']
    coordinates=np.asarray([r['box'] for r in rows],dtype=np.float64).copy()
    if settings['xywh']:coordinates[:,2:]+=coordinates[:,:2]
    ratio=640./max(w,h)
    padding=np.asarray([(640-w*ratio)/2,(640-h*ratio)/2]*2) if settings['center'] else np.zeros(4)
    coordinates=(coordinates-padding)/ratio
    if settings['clip']:coordinates=np.clip(coordinates,np.zeros(4),[w,h,w,h])
    scores=np.asarray([r['score'] for r in rows]);order=np.argsort(-scores,kind='stable');keep=[]
    for index in order.tolist():
        row=rows[index]
        if not row['enabled'] or row['score']<settings['confidence']:continue
        same=[j for j in keep if rows[j]['class_id']==row['class_id']]
        if same:
            prior=coordinates[same];box=coordinates[index]
            intersection=np.prod(np.maximum(0,np.minimum(prior[:,2:],box[2:])-np.maximum(prior[:,:2],box[:2])),axis=1)
            prior_area=np.prod(np.maximum(0,prior[:,2:]-prior[:,:2]),axis=1)
            area=np.prod(np.maximum(0,box[2:]-box[:2]));union=prior_area+area-intersection
            overlap=np.divide(intersection,union,out=np.zeros_like(intersection),where=union>0)
            if np.any((union>0)&(overlap>settings['nms'])):continue
        keep.append(index)
    result=[]
    for index in sorted(keep,key=lambda i:rows[i]['id']):
        row=rows[index];defect=(row['class_id']==0) if settings['swap'] else (row['class_id']!=0)
        result.append({'id':row['id'],'box':coordinates[index].tolist(),'label':'defect' if defect else 'part'})
    return {'objects':result}
