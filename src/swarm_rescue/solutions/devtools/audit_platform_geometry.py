"""Check platform archive geometry without loading pickle metadata."""
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from swarm_rescue.solutions.devtools.evaluate_verified_revision import ROOT


def main():
    for path in sorted((ROOT / '平台评测结果').glob('*.zip')):
        with ZipFile(path) as archive:
            for layout in sorted(n for n in archive.namelist() if n.endswith('_bombs.json')):
                folder = layout.rsplit('/', 1)[0]
                data = json.loads(archive.read(layout))
                with np.load(io.BytesIO(archive.read(folder + '/run.npz')), allow_pickle=False) as replay:
                    initial = replay['drone_states'][0]
                    starts = np.column_stack((initial['x'], initial['y']))
                image = next(n for n in archive.namelist() if n.startswith(folder + '/') and n.endswith('_wall_destruction.png'))
                print(path.name, data['map_name'], layout.split('/')[-1],
                      'start_xy', starts.round().astype(int).tolist(),
                      'wall_image', Image.open(io.BytesIO(archive.read(image))).size)


if __name__ == '__main__':
    main()
