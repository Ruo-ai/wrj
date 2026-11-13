from pathlib import Path
base=Path(__file__).parent/'optimization/evaluation'
p=base/'runner.py';s=p.read_text(encoding='utf-8-sig')
a=s.index('    dataset = (args.dataset');b=s.index('    for controller in args.controllers:',a)
s=s[:a]+'''    from .scenarios import official_rows,scenario,engine_profile
    official=getattr(args,'official_maps',None)
    if official:
        dataset=None;rows=official_rows(official)
        envelope={'dataset_id':'official-semifinal'}
        manifest_hash=digest(rows)
    else:
        dataset = (args.dataset or latest_output()).resolve()
        verify_dataset(dataset)
        envelope = read(dataset/'dataset.json')
        index = {r['id']:r for r in read(dataset/'index.json')['maps']}
        rows = [index[r['id']] for r in read(dataset/(args.split+'.json'))['maps']]
        manifest_hash=file_hash(dataset/'dataset.json')
    if args.limit is not None: rows=rows[:args.limit]
    if not rows: raise ValueError('Selected dataset split is empty')
    scenarios={r['id']:scenario(resolve_path(r['map_path']),r.get('official_map')) for r in rows}
    profile=engine_profile()
'''+s[b:]
s=s.replace("config = dict(dataset=path_name(dataset)","config = dict(protocol='semifinal-r0-v1',engine_profile=profile,scenarios=scenarios,dataset=path_name(dataset) if dataset else None")
s=s.replace("dataset_manifest_sha256=file_hash(dataset/'dataset.json')","dataset_manifest_sha256=manifest_hash")
s=s.replace("map='Generated'+row['geometry_sha256']","map=row.get('official_map') or 'Generated'+row['geometry_sha256'],official_map=row.get('official_map'),scenario=scenarios[row['id']]")
s=s.replace("    request = dict(map=item['map']", "    conditions=item.get('scenario',{'stage':'preliminary','zones':[]})\n    count=10 if phase=='blue' and conditions['stage']=='semifinal' else 5\n    request = dict(official_map=item.get('official_map'),stage=conditions['stage'],zones=conditions['zones'],map=item['map']")
s=s.replace("drones=5,bombs=2,recording=True","drones=count,bombs=2,recording=True")
s=s.replace("            from ..map_generator.official_format import read_spec\n            spec = read_spec(resolve_path(item['map_file']))\n            x,y = spec['spawn'];w,h = spec['size']", "            from .scenarios import scenario\n            spec=scenario(resolve_path(item['map_file']),item.get('official_map'))\n            x,y = spec['spawn'];w,h = spec['size']")
s=s.replace("zones_config=[],bombs=[dict", "zones_config=request['zones'],bombs=[dict")
p.write_text(s,encoding='utf-8')
p=base/'worker.py';s=p.read_text(encoding='utf-8-sig');s=s.replace("    cls = register(request['map_file'])",'''    from .scenarios import resolve_map,scenario,engine_profile,verify_recording
    conditions=scenario(request['map_file'],request.get('official_map'))
    expected_count=10 if phase=='blue' and conditions['stage']=='semifinal' else 5
    if request['drones']!=expected_count or request['zones']!=conditions['zones']:
        raise ValueError('Requested drones/zones differ from scenario contract')
    profile=engine_profile()
    cls = resolve_map(request)''')
s=s.replace("    telemetry = install_telemetry",'''    original_init=cls.__init__
    actual={}
    def checked_init(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        from swarm_rescue.simulation.elements.sensor_disablers import NoComZone,NoGpsZone
        actual.update(drones=len(self.drones),zones=sorted(z.name for z in self._zones_config),
            max_steps=self._max_timestep_limit,max_walltime=self._max_walltime_limit,
            instantiated_zone_types=sorted({('NO_COM_ZONE' if isinstance(e,NoComZone) else 'NO_GPS_ZONE')
                for e in self.playground.elements if isinstance(e,(NoComZone,NoGpsZone))}))
        if actual['drones']!=expected_count or actual['zones']!=request['zones'] or actual['instantiated_zone_types']!=request['zones']:
            raise ValueError('Native map conditions differ from requested scenario')
        if not request.get('smoke') and conditions['stage']=='semifinal' and request.get('official_map') in (None,'Map06','Map07t4','Map08','Map09'):
            if (actual['max_steps'],actual['max_walltime'])!=(5400,180):raise ValueError('Unexpected semifinal map limits')
    cls.__init__=checked_init
    telemetry = install_telemetry''')
s=s.replace("        result.update(phase=phase,", "        recordings=list((Path(request['engine_dir'])/'recordings').glob('*.npz'))\n        if result.get('status')=='ok':\n            if len(recordings)!=1: raise ValueError('Expected one recording')\n            verify_recording(recordings[0],request)\n        result.update(actual_conditions=actual,engine_profile=profile,phase=phase,")
p.write_text(s,encoding='utf-8')
p=base/'replay.py';s=p.read_text(encoding='utf-8-sig');s=s.replace("    cls = register(request['map_file'])\n    run_replay(str(recordings[0]),cls)","    from .scenarios import resolve_map,verify_recording\n    from swarm_rescue.simulation.elements.sensor_disablers import ZoneType\n    verify_recording(recordings[0],request)\n    cls = resolve_map(request)\n    run_replay(str(recordings[0]),cls,{'zones_config':tuple(ZoneType[n] for n in request['zones'])})")
p.write_text(s,encoding='utf-8')
p=base/'report.py';s=p.read_text(encoding='utf-8-sig');s=s.replace("    spec = read_spec(resolve_path(pair['map_file']))",'''    if pair.get('official_map'):
        import json
        from swarm_rescue.tools.map_wall_editor import load_elements,render_elements_to_image
        import cv2
        doc=json.loads(resolve_path(pair['map_file']).read_text(encoding='utf-8'))
        w,h=doc['canvas']['width'],doc['canvas']['height']
        background=cv2.cvtColor(render_elements_to_image(w,h,load_elements(doc,w,h)),cv2.COLOR_BGR2RGB)
        spec={'size':[w,h],'walls':[],'center':[-100,-100]}
    else:
        spec = read_spec(resolve_path(pair['map_file']))
        background=None''')
s=s.replace("    for _,l,t,r,b in spec['walls']:","    if background is not None: ax.imshow(background,extent=(0,w,h,0))\n    for zone in spec.get('zones',[]):\n        l,t,r,b=zone['rect'];ax.add_patch(Rectangle((l,t),r-l,b-t,color='gold' if zone['type']=='NO_COM_ZONE' else 'gray',alpha=.18))\n    for _,l,t,r,b in spec['walls']:")
p.write_text(s,encoding='utf-8')
p=Path(__file__).parent/'optimization/dataset_manager/index.py';s=p.read_text(encoding='utf-8-sig');s=s.replace("(r['room_count'] for r in rows)","(r['room_count'] for r in rows if r['room_count'] is not None)");p.write_text(s,encoding='utf-8')
