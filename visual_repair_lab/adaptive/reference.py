"""Independent state reducers for the declared hook semantics.

No generated source or compiler helper is imported. Shared data schemas and base
parameter tables remain part of the specification, not independent evidence.
"""
import copy
from vision_sort.visual_training.tasks import parameters


def enabled(rule, family, **v):
    c=rule['condition']
    if c=='always':return True
    if family=='drag_hysteresis':
        return {'wide':v['layout']=='wide','compact':v['layout']=='compact','zoom_high':v['zoom']>1.5,'range':1<=v['zoom']<=1.5}[c]
    if family=='linked_brush':
        return {'view_a':v['view']=='A','view_b':v['view']=='B','zoom_high':v['zoom']>1,'range':150<=v['x']<=450}[c]
    return {'positive':v['vy']>0,'negative':v['vy']<0,'gravity_positive':v['gravity']>0,'range':100<=abs(v['vy'])<=600}[c]


def reference(spec,state,disabled=()):
    rules=[r for i,r in enumerate(spec['rules']) if i not in disabled]
    family=spec['family'];mask=spec['world']['mask'];bits=[bool(mask&(1<<i)) for i in range(6)]
    args=parameters(family,mask,'train')
    return {'drag_hysteresis':drag,'linked_brush':chart,'contact_dynamics':physics}[family](copy.deepcopy(state),bits,args,rules)


def drag(s,b,args,rules):
    lift,scales,grid=args;offset={k:[0.,0.] for k in s['ids']};width,zoom,layout=640,1.,'compact'
    active=None;origin=[0.,0.];start=[0.,0.]
    def applies(r):return enabled(r,'drag_hysteresis',layout=layout,zoom=zoom)
    def positions():
        out={}
        for i,key in enumerate(s['ids']):
            x=(60+(i%2)*140 if layout=='compact' else 40+i*130)+offset[key][0]
            y=(220+(i//2)*110 if layout=='compact' else 270)+offset[key][1]
            for r in reversed(rules):
                if r['op']=='limit' and applies(r):x=min(width-90-r['value'],max(r['value'],x))
            out[key]=[x,y,90,70]
        return out
    for e in [{'kind':'initial'}]+s['events']:
        kind=e['kind']
        if kind=='reset_scene':width,zoom,layout=640,1.,'compact';offset={k:[0.,0.] for k in s['ids']};active=None
        if kind=='resize':
            width=e['width'];high=520 if b[0] else 480
            for r in rules:
                if r['op']=='threshold' and applies(r):high=r['value']
            low=high-(40 if b[1] else 0)
            if layout=='compact' and width>high:layout='wide'
            elif layout=='wide' and width<low:layout='compact'
        if kind=='set_zoom':zoom=e['zoom']
        p=positions()
        if kind=='down':
            active=None
            for key in s['ids'][::-1]:
                x,y,w,h=p[key]
                if x<=e['x']<=x+w and y<=e['y']<=y+h:active=key;origin=[e['x'],e['y']];start=offset[key][:];break
        if kind=='move' and active is not None:
            x,y,w,h=p[active]
            if not b[2] and not(x<=e['x']<=x+w and y<=e['y']<=y+h):active=None
            else:
                factor=1.
                for r in rules:
                    if r['op']=='gain' and applies(r):factor*=r['value']
                divisor=zoom if b[5] else 1
                offset[active]=[start[0]+(e['x']-origin[0])/divisor*factor,start[1]+(0 if b[4] else (e['y']-origin[1])/divisor*factor)]
            if active is not None:
                for r in rules:
                    if r['op']=='order' and applies(r):offset[active][0]=round(offset[active][0]*zoom/r['value'])*r['value']/zoom
        if kind=='up' and active is not None:
            for r in reversed(rules):
                if r['op']=='snap' and applies(r):offset[active]=[round(v/r['value'])*r['value'] for v in offset[active]]
            if b[3]:offset[active]=[round(v/grid)*grid for v in offset[active]]
            active=None
        p=positions()
    if active is not None:
        p[active][1]-=lift*(zoom if scales else 1)
        for r in rules:
            if r['op']=='offset' and applies(r):p[active][1]-=r['value']
    return {'positions':p,'active_id':active,'layout':layout}


def segment(a,b,box):
    low,high=0.,1.
    for axis in (0,1):
        d=b[axis]-a[axis]
        if d==0:
            if not box[axis]<=a[axis]<=box[axis+2]:return False
        else:
            t0,t1=(box[axis]-a[axis])/d,(box[axis+2]-a[axis])/d
            low=max(low,min(t0,t1));high=min(high,max(t0,t1))
            if low>high:return False
    return True


def chart(s,b,args,rules):
    lo,hi,pan_scales=args
    def fresh():return dict(context='overview',zoom=1.,pan_x=0.,pan_y=0.,series=s['series'])
    frame=fresh();brushes=[]
    def project(f):
        power,rev=1,False
        if f['context']!='overview':
            off=0 if f['context']=='A' else 2;power=2 if b[off] else 1;rev=b[off+1]
        z=f['zoom'];out={}
        for key,values in f['series'].items():
            out[key]=[]
            for a,c in values:
                x=70+460*z*a**power+f['pan_x']*(z if pan_scales else 1)
                y=200+300*z*(c if rev else 1-c)+f['pan_y']*(z if pan_scales else 1)
                for r in rules:
                    if not enabled(r,'linked_brush',view=f['context'],zoom=z,x=x):continue
                    op,v=r['op'],r['value']
                    if op=='limit':x=min(v,max(70,x))
                    elif op=='gain':x=70+(x-70)*v
                    elif op=='threshold':x=70 if x<v else x
                    elif op=='snap':x=round(x/v)*v
                    elif op=='offset':y+=v
                    else:x=70+round((x-70)/z/v)*v*z
                out[key].append([x,y])
        return out
    for e in s['events']:
        kind=e['kind']
        if kind=='reset_scene':frame=fresh();brushes=[]
        elif kind=='set_view':frame['context']=e['context']
        elif kind=='set_transform':frame.update({k:e[k] for k in ('zoom','pan_x','pan_y')});frame['zoom']=min(hi,max(lo,frame['zoom']))
        elif kind=='new_data':frame['series']=e['series']
        elif kind=='clear_selection':brushes=[]
        elif kind=='brush':brushes.append((frame.copy(),[min(e['x0'],e['x1']),min(e['y0'],e['y1']),max(e['x0'],e['x1']),max(e['y0'],e['y1'])],e['mode']))
    selected=set()
    for old,box,mode in brushes:
        hits=set()
        for key,line in project(old if b[5] else frame).items():
            if any(box[0]<=x<=box[2] and box[1]<=y<=box[3] for x,y in line) or b[4] and any(segment(a,c,box) for a,c in zip(line,line[1:])):hits.add(key)
        if mode=='replace':selected=hits
        elif mode=='add':selected|=hits
        elif mode=='subtract':selected-=hits
        else:selected^=hits
    return dict(projected=project(frame),selected_ids=sorted(selected),linked_counts=[int(k in selected) for k in sorted(s['series'])])


def physics(s,b,args,rules):
    negative,positive,before=args;body=s['initial_body'].copy();gravity,dt,count,grounded,contact=0.,1/60,0,False,None
    floor,wall=s['floor'],s['wall']
    def applies(r):return enabled(r,'contact_dynamics',vy=body['vy'],gravity=gravity)
    for e in s['events']:
        kind=e['kind']
        if kind=='reset_scene':body=s['initial_body'].copy();gravity,dt,count,grounded,contact=0.,1/60,0,False,None
        elif kind=='set_body':body.update({k:e[k] for k in ('x','y','vx','vy')});grounded,contact=False,None
        elif kind=='set_physics':gravity,dt=e['gravity'],e['dt']
        elif kind=='advance':
            for _ in range(e['ticks']):
                grounded,contact=False,None
                step_before=before
                for r in rules:
                    if r['op']=='order' and applies(r):step_before=r['value']
                if step_before:body['vy']=min(positive,max(-negative,body['vy']))
                g=gravity*(1.5 if b[4] else 1)*dt;body['vy']+=g
                for r in rules:
                    if r['op']=='gain' and applies(r):body['vy']+=g*(r['value']-1)
                if not step_before:body['vy']=min(positive,max(-negative,body['vy']))
                for r in rules:
                    if applies(r):
                        if r['op']=='limit':body['vx']=min(r['value'],max(-r['value'],body['vx']))
                        if r['op']=='threshold' and 0<body['vy']<r['value']:body['vy']=0
                body['vx']*=max(0,1-(.04 if b[3] else 0)*dt*60)
                nx,ny=body['x']+body['vx']*dt,body['y']+body['vy']*dt
                for r in reversed(rules):
                    if applies(r):
                        if r['op']=='snap':nx=round(nx/r['value'])*r['value']
                        if r['op']=='offset':ny+=r['value']
                if body['vx']>0 and body['x']+body['width']<=wall['x']<=nx+body['width'] and body['y']+body['height']>wall['y'] and body['y']<wall['y']+wall['height']:
                    nx=wall['x']-body['width'];body['vx']*=-int(b[5]);count+=1;contact='wall'
                overlap=nx+body['width']>floor['x'] and nx<floor['x']+floor['width']
                if overlap and body['vy']>0 and body['y']+body['height']<=floor['y']<=ny+body['height']:
                    ny=floor['y']-body['height'];body['vy']*=-(.6 if b[1] else 0);body['vx']*=1-(.3 if b[2] else 0);count+=1;contact,grounded='floor',body['vy']==0
                elif overlap and not b[0] and body['vy']<0 and body['y']>=floor['y']+floor['height']>=ny:
                    ny=floor['y']+floor['height'];body['vy']=abs(body['vy'])*(.6 if b[1] else 0);count+=1;contact='floor'
                if overlap and body['vy']==0 and abs(ny+body['height']-floor['y'])<1e-7:grounded,contact=True,'floor'
                body['x'],body['y']=nx,ny
    return {k:body[k] for k in ('x','y','vx','vy')}|dict(grounded=grounded,contact=contact,contacts=count)
