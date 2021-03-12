import numpy as np
from torch.utils.data.dataset import Dataset


class ClassificationTask(Dataset):
    def __init__(self, task='yin-yang', r_small=0.1, r_big=0.5, size=1000, seed=42):
        super(ClassificationTask, self).__init__()
        # using the numpy RNG to allow compatibility to other deep learning frameworks
        np.random.seed(seed)
        self.r_small = r_small
        self.r_big = r_big
        self.__vals = []
        self.__cs = []
        self.task = task
        if self.task in ['xor', 'circles', 'linear', 'wave']:
            self.class_names = ['blue', 'orange']
        elif self.task in ['yin-yang', 'dots']:
            self.class_names = ['yin', 'yang', 'dot']
        elif self.task == 'squares':
            self.class_names = ['blue', 'orange', 'green', 'red']
        else:
            raise NotImplementedError('No known task specified...')

        for i in range(size):
            # keep num of class instances balanced by using rejection sampling
            # choose class for this sample
            goal_class = np.random.randint(len(self.class_names))
            x, y, c = self.get_sample(goal=goal_class)
            # add mirrod axis values
            x_flipped = 1. - x
            y_flipped = 1. - y
            val = np.array([x, y, x_flipped, y_flipped])
            self.__vals.append(val)
            self.__cs.append(c)

    def get_sample(self, goal=None):
        # sample until goal is satisfied
        found_sample_yet = False
        while not found_sample_yet:
            # sample x,y coordinates
            x, y = np.random.rand(2) * 2. * self.r_big

            # check if within yin-yang circle
            if self.task == 'yin-yang' and np.sqrt((x - self.r_big)**2 + (y - self.r_big)**2) > self.r_big:
                continue

            # check if they have the same class as the goal for this sample
            c = self.which_class(x, y)
            if goal is None or bool(c[goal]):
                found_sample_yet = True
                break
        return x, y, c

    def which_class(self, x, y):
        if self.task == 'yin-yang':
            # equations inspired by
            # https://link.springer.com/content/pdf/10.1007/11564126_19.pdf
            d_right = self.dist_to_right_dot(x, y)
            d_left = self.dist_to_left_dot(x, y)
            criterion1 = d_right <= self.r_small
            criterion2 = d_left > self.r_small and d_left <= 0.5 * self.r_big
            criterion3 = y > self.r_big and d_right > 0.5 * self.r_big
            is_yin = criterion1 or criterion2 or criterion3
            is_circles = d_right < self.r_small or d_left < self.r_small
            if is_circles:
                return np.array([0, 0, 1])
            elif is_yin:
                return np.array([0, 1, 0])
            else:
                return np.array([1, 0, 0])

        elif self.task == 'xor':
            if abs(y) > 0.5 and abs(x) > 0.5 or abs(y) < 0.5 and abs(x) < 0.5:
                return np.array([1, 0, 0, 0])
            else:
                return np.array([0, 1, 0, 0])

        elif self.task == 'wave':
            d_right = self.dist_to_right_dot(x, y)
            d_left = self.dist_to_left_dot(x, y)
            criterion1 = d_left < self.r_small
            criterion2 = d_left > self.r_small and d_left <= 0.5 * self.r_big
            criterion3 = y > self.r_big and d_right > 0.5 * self.r_big
            if criterion1 or criterion2 or criterion3:
                return np.array([0, 1, 0, 0])
            else:
                return np.array([1, 0, 0, 0])

        elif self.task == 'squares':
            if x > 0.5 and y > 0.5:
                return np.array([0, 0, 0, 1])
            elif y > 0.5:
                return np.array([0, 0, 1, 0])
            elif x < 0.5:
                return np.array([0, 1, 0, 0])
            else:
                return np.array([1, 0, 0, 0])

        elif self.task == 'circles':
            radius = np.sqrt((x-0.5)**2 + (y-0.5)**2)
            if  radius < 0.3:
                return np.array([1, 0, 0, 0])
            else:
                return np.array([0, 1, 0, 0])

        elif self.task == 'dots':
            r1 = np.sqrt((x-0.3)**2 + (y-0.3)**2)
            r2 = np.sqrt((x-0.7)**2 + (y-0.7)**2)
            # r1 = np.sqrt((x-0.)**2 + (y-0.)**2)
            # r2 = np.sqrt((x-1.)**2 + (y-1.)**2)
            if r1 < self.r_small:
                return np.array([1, 0, 0, 0])
            elif r2 < self.r_small:
                return np.array([0, 1, 0, 0])
            else:
                return np.array([0, 0, 1, 0])

        elif self.task == 'linear':
            if y > 0.5:
                return np.array([1, 0, 0, 0])
            else:
                return np.array([0, 1, 0, 0])

        else:
            raise NotImplementedError('No known task specified...')

    def dist_to_right_dot(self, x, y):
        return np.sqrt((x - 1.5 * self.r_big)**2 + (y - self.r_big)**2)

    def dist_to_left_dot(self, x, y):
        return np.sqrt((x - 0.5 * self.r_big)**2 + (y - self.r_big)**2)

    def __getitem__(self, index):
        return self.__vals[index], self.__cs[index]

    def __len__(self):
        return len(self.__cs)
