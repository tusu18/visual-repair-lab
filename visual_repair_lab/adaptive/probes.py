"""Public control palette. It contains no targets, grading outcomes or task IDs."""
from vision_sort.visual_training import tasks


def experiments(family):
    e=[]
    if family=='drag_hysteresis':
        for width in (280,440,459,461,480,500,520,539,541,640):
            e.append([dict(kind='resize',width=width)])
        for width in (440,640):
            for zoom in (1.,2.):
                for delta in (14,36):
                    x,y=(80,250) if width==440 else (60,290)
                    e.append([dict(kind='resize',width=width),dict(kind='set_zoom',zoom=zoom),
                        dict(kind='down',x=x,y=y),dict(kind='move',x=x+delta,y=y+14),dict(kind='up')])
                    e.append(e[-1][:-1])
        e += [[dict(kind='down',x=80,y=250)],
              [dict(kind='set_zoom',zoom=2.),dict(kind='down',x=80,y=250)]]
    elif family=='linked_brush':
        for view in ('overview','A','B'):
            for zoom in (.5,1.,2.):
                e.append([dict(kind='set_view',context=view),dict(kind='set_transform',zoom=zoom,pan_x=35,pan_y=-25)])
                e.append([dict(kind='set_view',context=view),dict(kind='set_transform',zoom=zoom,pan_x=-90,pan_y=100)])
        e += [[dict(kind='brush',x0=140,x1=450,y0=220,y1=470,mode='replace')],
              [dict(kind='set_view',context='A'),dict(kind='brush',x0=60,x1=260,y0=180,y1=420,mode='replace')]]
    else:
        for vy in (-500,-40,10,50,400,700):
            for gravity in (0,800):
                e.append([dict(kind='set_body',x=203,y=150,vx=310,vy=vy),
                    dict(kind='set_physics',gravity=gravity,dt=.025),dict(kind='advance',ticks=2)])
        e += [[dict(kind='set_body',x=200,y=300,vx=100,vy=800),dict(kind='advance',ticks=3)],
              [dict(kind='set_body',x=203,y=160,vx=-320,vy=-500),dict(kind='advance',ticks=2)]]
    return e


def state_for(family,seed,events):
    state=tasks.initial_state(family,seed)
    for event in events:
        state=tasks.transition(family,state,event)
    return state


def grading_states(spec,pool):
    family=spec['family'];seed=spec['scene_seed']
    # Separate seeds and distinct prefix events keep exact progress/final states
    # disjoint. Shared mechanisms remain intentional correctness supervision.
    cases=tasks.protected_states(family,seed,'train',pool=pool)
    prefix=[] if pool=='final' else [dict(kind='reset_scene')]
    cases += [state_for(family,seed,prefix+sequence) for sequence in experiments(family)]
    return cases
