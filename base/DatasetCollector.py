import os
import logging
from ipdb import set_trace
import numpy as np

class DatasetCollector:

    def __init__(self):
        pass

    def classes(self):
        return []

    def train(self, cls=None):
        return []

    def val(self, cls=None):
        return []

    def test(self, cls=None):
        return []


class SanityCollector(DatasetCollector):

    def __init__(self, *args, **kwargs):
        self.cls = ['chair']

    def classes(self):
        return self.cls

    def _gather(self):
        return [('./data/model.128.png', './data/model.shl.mat')]

    def train(self, cls=None):
        return self._gather()

    def val(self, cls=None):
        return self._gather()

    def test(self, cls=None):
        return self._gather()


class ShapeNetPTNCollector(DatasetCollector):
    def __init__(self, base_dir, crop=True):
        assert os.path.exists(base_dir), ('Base directory for PTN dataset does not exist [%s].' % base_dir)
        self.base_dir  = base_dir
        self.id_dir    = os.path.join(self.base_dir, 'shapenetcore_ids')
        self.view_dir  = os.path.join(self.base_dir, 'shapenetcore_viewdata')
        self.shape_dir = os.path.join(self.base_dir, 'shapenetcore_voxdata')
        self.crop      = crop
        self.cls       = []

        for c in sorted([d[:-12] for d in os.listdir(self.id_dir) if d.endswith('_testids.txt')]):
            if  os.path.exists(os.path.join(self.id_dir, c+'_trainids.txt')) and \
                os.path.exists(os.path.join(self.id_dir, c+'_valids.txt')) and \
                os.path.exists(os.path.join(self.view_dir, c)) and \
                os.path.exists(os.path.join(self.shape_dir, c)):
                self.cls.append(c)
                pass
            pass
        pass

    def _gather(self, subset, cls=None):
        if cls is None:
            cls = self.classes()
            pass

        samples = []    

        shape_suffix = 'model.shl.mat' if self.representation == 'shl' else 'model.vox.mat'
        for c in cls:
            logging.info('Collecting %s/%s...' % (subset, c))
            with open(os.path.join(self.id_dir, '%s_%sids.txt' % (c, 'all'))) as f:
                for line in f:
                    id = line.strip().split('/')[1]
                    shapepath = os.path.join(self.shape_dir, c, id, shape_suffix)
                    viewdir = os.path.join(self.view_dir, c, id)                    
                    for file in sorted(os.listdir(viewdir)):
                        if self.crop and file.endswith('.128.png'):
                            samples.append((os.path.join(viewdir, file), shapepath))
                            pass
                        if not self.crop and file.endswith('.png') and not file.endswith('.128.png'):
                            samples.append((os.path.join(viewdir, file, id), shapepath))
                            pass
                        pass
                    pass
                pass
            pass

        return samples

    def classes(self):
        return self.cls

    def train(self, cls=None):
        return self._gather('train', cls)

    def val(self, cls=None):
        return self._gather('val', cls)

    def test(self, cls=None):
        return self._gather('test', cls)
    pass


class BlendswapOGNCollector(DatasetCollector):

    def __init__(self, base_dir, resolution=512):
        res2dir = {64:'64_l4', 128:'128_l4', 256:'256_l5', 512:'512_l5'}
        self.base_dir = os.path.join(base_dir, res2dir[resolution])
        assert os.path.exists(self.base_dir), ('Base directory for OGN Blendswap dataset does not exist [%s].' % self.base_dir)
        pass

    def _gather(self):
        samples = []
        shape_suffix = '.shl.mat'
        
        for file in sorted(os.listdir(self.base_dir)):
            if file.endswith(shape_suffix):                
                samples.append(os.path.join(self.base_dir, file))
                pass
            pass

        return samples

    def classes(self):
        return None

    def train(self):
        return self._gather('all')

    def val(self):
        return self._gather('all')

    def test(self):
        return self._gather('all')
    pass


class ShapeNetCarsOGNCollector(DatasetCollector):
    def __init__(self, base_dir, shapenet_base_dir, resolution=128, crop=True):
        res2dir = {64:'64_l4', 128:'128_l4', 256:'256_l4'}
        self.base_dir = os.path.join(base_dir, res2dir[resolution])        
        assert os.path.exists(self.base_dir), ('Base directory for OGN ShapeNet Cars dataset does not exist [%s].' % self.base_dir)
        self.shapenet_base_dir = shapenet_base_dir
        assert os.path.exists(self.shapenet_base_dir), ('ShapeNet rendering directory for OGN ShapeNet Cars dataset does not exist [%s].' % self.shapenet_base_dir)

        self.crop = crop
        
        for s in ['train', 'validation', 'test']:
            id_path = os.path.join(self.base_dir, 'shapenet_cars_rendered_new_%s.txt' % s)
            assert os.path.exists(id_path), ('Could not find id list for %s set [%s].' % (s, id_path))
            pass
        
        assert os.path.exists(self.base_dir), ('Base directory for OGN ShapeNet Cars dataset does not exist [%s].' % self.base_dir)
        pass

    def classes(self):
        return ['car']

    def _gather(self, subset):
        samples = []
        with open(os.path.join(self.base_dir, 'shapenet_cars_rendered_new_%s.txt' % subset)) as f:
            for line in f:
                img_path, id = line.strip().split(' ')
                img_id      = img_path.split('/')[-1]
                shapenet_id = img_path.split('/')[-3]
                img_path    = os.path.join(self.shapenet_base_dir, '02958343', shapenet_id, \
                    'rendering', img_id + ('.128.png' if self.crop else '.png'))
                shape_path  = os.path.join(self.base_dir, id + shape_suffix)
                samples.append((img_path, shape_path))
                pass
            pass
        return samples

    def train(self, cls=None):
        return self._gather('train')

    def val(self, cls=None):
        return self._gather('validation')

    def test(self, cls=None):
        return self._gather('test')
    pass


class ShapeNet3DR2N2Collector(DatasetCollector):
    def __init__(self, base_dir, cat_id, representation='vox', side=32, p_norm=2, k_shot='full',
            rand_mode='False',  snMini=False, snMini_size=50, snMini_mode='random'):
        assert representation in ['vox', 'sdf', 'chamfer']
        if representation == 'vox':
            self.shape_dir = os.path.join(base_dir, 'vox_orig')

        elif representation == 'sdf':
            self.shape_dir = os.path.join(base_dir, 'sdfs')
        elif representation == 'chamfer':
            self.shape_dir = os.path.join(base_dir, 'ShapeNetDist'+str(side))
        else:
            print('unknown representation')
            exit()
        self.view_dir  = os.path.join(base_dir, 'ShapeNetRendering')
        if k_shot == 'full':
            self.list_dir = os.path.join(base_dir, 'ids')
        else:
            self.list_dir = os.path.join(base_dir, 'ids'+'_fs',k_shot)

        if snMini:
            if snMini_mode == 'cluster':
                self.list_dir = os.path.join(base_dir, 'ids'+'_ShapeNetMini',
                        snMini_mode, 'kmenoids', str(snMini_size))
            else:
                self.list_dir = os.path.join(base_dir, 'ids'+'_ShapeNetMini',
                        snMini_mode, str(snMini_size))

        self.representation = representation
        self.side = side
        self.p_norm = p_norm
        self.k_shot = k_shot
        self.rand_mode = rand_mode
        self.snMini = snMini
        self.snMini_size = snMini_size 
        self.snMini_mode = snMini_mode 
        if not os.path.exists(self.list_dir) or 1==2: # FIXED
            import sys
            sys.path.append('./external/')
            from generate3DR2N2split import write_split
            write_split(base_dir)
            pass

        self.cls = []
        if k_shot == 'full': # we take all 13 categories
            self.cls = ['02691156', '02933112', '03001627', '03636649', '04090263', '04379243',
                '04530566', '02828884', '02958343', '03211117', '03691459', '04256520',
                '04401088', '03624134', '03467517', '03642806', '02808440']

        elif k_shot == '0': # we only take 7 base categories
            self.cls = ['02691156', '02958343', '03001627', '03211117', '04401088', '03691459',
                '04379243']
        else: # we take 6 novel categories
            self.cls = ['02828884', '02933112', '03636649', '04090263', '04256520', '04530566',
                    '03624134', '03467517', '03642806', '02808440']


        # create simple ids for classes 
        class_id_dic = {'02691156' : 0,
                '02958343': 1,
                '03001627': 2,
                '03211117': 3,
                '04401088': 4,
                '03691459': 5,
                '04379243': 6,
                '02828884': 7,
                '02933112': 8,
                '03636649': 9,
                '04090263': 10,
                '04256520': 11,
                '04530566': 12,
                '03624134': 13,
                '03467517': 14,
                '03642806': 15,
                '02808440': 16
                }
        self.cls_dic = class_id_dic

        

    def classes(self):
        return self.cls

    def _gather(self, subset, cls=None):
        if cls is None:
            cls = self.classes()
            pass
        samples = []    
        if self.representation == 'vox':
             shape_suffix = 'model.binvox'
        elif self.representation == 'sdf':
            shape_suffix = 'sdf_manifold_50000_grid_size_32.npy'
        elif self.representation == 'chamfer':
            shape_suffix\
                = 'actual_dist_grid_size_'+str(self.side)+'_p_'+str(self.p_norm)+'.npy'

        else:
            print('unknown shape suffix')
            exit()

        for c in cls:
            logging.info('Collecting %s/%s...' % (subset, c))

            with open(os.path.join(self.list_dir,c, '%s_%s.txt' % (c, subset))) as f:
                for line in f:
                    # format is class/id
                    id = line.strip()
                    if self.representation == 'vox':
                        shapepath = os.path.join(self.shape_dir, c, str(self.side), id,
                            id+'_vox_'+str(self.side)+'.npy')
                    else:
                        shapepath = os.path.join(self.shape_dir, c, str(self.side), id,
                            id+'_sdf_'+str(self.side)+'_pad_1.npy')
                    # check images
                    viewdir = os.path.join(self.view_dir, c, id, 'rendering')
                    tejzzz_sup = os.listdir(viewdir)
                    np.random.shuffle(tejzzz_sup)
                    if 'final_views' in self.snMini_mode\
                            or 'final_full_views' in self.snMini_mode: # take a random subset of views
                                views_prun_nr = int(self.snMini_mode[self.snMini_mode.rfind('_')+1:])
                    else:
                        views_prun_nr = 24
                    views_prun_ctr = 0

                    for file in sorted(os.listdir(viewdir)):
                        if file.endswith('.128.png') and (not file.endswith('128.128.png')): # FIX
                            if self.rand_mode == 'False':
                                samples.append((os.path.join(viewdir, file), shapepath, self.cls_dic[c], id))
                                views_prun_ctr += 1
                                if views_prun_ctr >= views_prun_nr:
                                    break
                            elif self.rand_mode == 'base':
                                samples.append((os.path.join(viewdir, file), shapepath, np.random.randint(0,7), id))
                                views_prun_ctr += 1
                                if views_prun_ctr >= views_prun_nr:
                                    break

                            pass
                        if not file.endswith('.128.png'): # FIX
                            pass
                        pass
                    pass
                pass
            pass

        return samples

    def train(self, cls=None):
        return self._gather('train', cls)

    def val(self, cls=None):
        return []

    def test(self, cls=None):
        return self._gather('test', cls)        
    pass
