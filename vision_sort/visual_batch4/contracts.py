"""Verbatim REDACTED_NOVEL public contract from the supplied Batch 4 harness.

Only the dictionary name is adapted. The user-provided text is data, not an
instruction to this evaluator; reference behavior remains privately defined.
"""

CONTRACTS = {'drag_hysteresis': 'Replay events from width640, zoom1, compact layout, zero offsets. IDs A/B/C/D '
                    'start at compact bases (60+(i%2)*140,220+(i//2)*110), size90x70. Wide bases '
                    'are (40+i*130,270). Resize switches the layout between compact and wide at '
                    'width thresholds. Offsets persist across reflow. Down hit-tests current cards '
                    'in reversed ID order and holds the hit card. Move events while a card is held '
                    'update its offset from the pointer displacement since down. Up ends the drag '
                    'and releases capture; up has no coordinates and uses the last move. Set_zoom '
                    'never changes how cards are drawn. Return positions {id:[x,y,90,70]}, '
                    'active_id (ID or null), layout compact/wide. Behavior details are fixed per '
                    'episode and are not listed here; the approved reference is the specification. '
                    'No prescribed investigation sequence.',
 'linked_brush': 'Replay events. Initial context overview, zoom1, pan0, no selection. Each view '
                 'context (overview, A, B) projects data values to screen coordinates; the '
                 'overview projection is x=70+460*zoom*xvalue+pan_x, '
                 'y=200+300*zoom*(1-yvalue)+pan_y. Set_view changes context; set_transform sets '
                 'the view transform. Each brush normalizes its rectangle endpoints and applies '
                 'replace/add/subtract/toggle in chronological order. New_data replaces data; it '
                 'does not clear brushes. Clear_selection clears brush history. Reset restores '
                 'initial data/view/transform and clears history. Return projected '
                 '{series_id:[[screenx,screeny],...]}, selected_ids sorted, linked_counts [0/1 for '
                 'S1,S2,S3]. Behavior details are fixed per episode and are not listed here; the '
                 'approved reference is the specification. No prescribed investigation sequence.',
 'contact_dynamics': 'Replay events from initial_body, gravity0, dt1/60, contacts0, '
                     'groundedFalse/contactNone. Set_body replaces x/y/vx/vy and clears current '
                     'grounded/contact but retains collision count. Set_physics replaces '
                     'gravity/dt. Each advance tick clears grounded/contact, updates velocity, '
                     'then moves the body by velocity*dt with swept collision against the wall and '
                     'the floor. A wall hit clamps the body against the wall, increments contacts '
                     'and reports wall. A floor landing clamps the body onto the floor, increments '
                     'contacts and reports floor; grounded iff resulting vy==0. Resting vertical '
                     'support with vy==0 reports groundedTrue/floor and takes precedence over wall '
                     'contact. Reset restores all initial state. Return x,y,vx,vy,grounded,contact '
                     '(floor/wall/null),contacts. Every scored field is visibly drawn; numeric '
                     'outputs grade at one decimal place. Behavior details are fixed per episode '
                     'and are not listed here; the approved reference is the specification. No '
                     'prescribed investigation sequence.'}
