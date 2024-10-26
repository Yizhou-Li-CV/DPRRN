import numpy as np
import math
from numpy import linalg as LA

class RaindropRenderer:
    def __init__(self, params):
        self.n_water = 1.33
        self.n_air = 1.0
        self.gamma = math.asin(self.n_air / self.n_water)

        self.M = np.random.randint(*params['M_range'])
        # only two M: close and far
        # self.M = random.choice(*params['M_range'])

        self.B = np.random.randint(*params['B_range'])
        self.psi = np.float(np.random.randint(*params['psi_range'])) / 180.0 * np.pi
        self.tau = np.float(np.random.randint(*params['tau_range'])) / 180.0 * np.pi
        self.density_range = params['density_range']

        # vertical fov only. as no crop during rendering, vertical fov is for lens with focal len 27mm
        # but we only want center raindrops (1.5x), so we use fov for 1.5x focal length
        self.half_fov = 30 / 2 / 180.0 * np.pi

        self.aspect_ratio = 4 / 3  # when sensor is square, the top and bottom no need to change.
        # If not square, the top and bottom will be longer.
        self.visible_area_top = 2 * self.M * math.tan(self.psi) * math.tan(self.half_fov) / \
                                (math.tan(self.psi) + math.tan(self.half_fov)) * self.aspect_ratio
        self.visible_area_bottom = 2 * self.M * math.tan(self.psi) * math.tan(self.half_fov) / \
                                   (math.tan(self.psi) - math.tan(self.half_fov)) * self.aspect_ratio
        self.h1 = self.M * math.tan(self.half_fov) / (
                    math.cos(self.half_fov) * (math.tan(self.psi) + math.tan(self.half_fov)))
        self.h2 = self.M * math.tan(self.half_fov) / (
                    math.cos(self.half_fov) * (math.tan(self.psi) - math.tan(self.half_fov)))

        self.visible_area = (self.visible_area_top + self.visible_area_bottom) * (self.h1 + self.h2) / 2
        self.visible_area_cm2 = self.visible_area / 100

        print('visible area in cm2:', self.visible_area_cm2)

        self.max_raindrop_num = self.visible_area_cm2

        print(self.M, self.B, 'h1', self.h1, 'h2', self.h2, 'top', self.visible_area_top, 'area_cm2',
              self.visible_area_cm2)
        print(params)

        self.fx = params['fx']
        self.fy = params['fy']
        self.cx = params['cx']
        self.cy = params['cy']

        self.intrinsic = np.zeros([3, 3])
        self.intrinsic[0, 0] = self.fx
        self.intrinsic[1, 1] = self.fy
        self.intrinsic[0, 2] = self.cx
        self.intrinsic[1, 2] = self.cy
        self.intrinsic[2, 2] = 1

        self.normal = np.array([0, -1 * math.cos(self.psi), math.sin(self.psi)])
        self.o_g = (self.normal[2] * self.M / np.dot(self.normal, self.normal)) * self.normal

    def to_glass(self, x, y):
        w = self.M * math.tan(self.psi) / (math.tan(self.psi) - (y - self.intrinsic[1, 2]) / self.intrinsic[1, 1])
        u = w * (x - self.intrinsic[0, 2]) / self.intrinsic[0, 0]
        v = w * (y - self.intrinsic[1, 2]) / self.intrinsic[1, 1]

        return u, v, w

    def w_in_plane(self, u, v):
        return (self.normal[2] * self.M - self.normal[0] * u - self.normal[1] * v) / self.normal[2]

    def get_sphere_raindrop(self, W, H, mode=None):
        self.g_centers = []
        self.g_radius = []
        self.radius = []
        self.centers = []

        left_upper = self.to_glass(0, 0)
        left_bottom = self.to_glass(0, H)
        right_upper = self.to_glass(W, 0)
        right_bottom = self.to_glass(W, H)

        ns = [np.random.randint(150, 500), np.random.randint(500, 1300)]
        glass_r_ranges = [(0.8, 1.8), (0.2, 0.8)]

        nb_modes = len(glass_r_ranges)
        if mode is None:
            mode = np.random.randint(0, nb_modes)
            print('random mode:', mode)
        elif mode >= nb_modes or mode < 0:
            assert mode >= nb_modes, f'Error: mode >= {nb_modes} or mode < 0'
        else:
            print('current selected mode:', mode)
        n = ns[mode]
        glass_r_range = glass_r_ranges[mode]

        print('current raindrop num:', n, 'raindrop_range:', glass_r_range)
        for i in range(n):
            u = left_bottom[0] + (right_bottom[0] - left_bottom[0]) * np.random.random_sample()
            v = left_upper[1] + (right_bottom[1] - left_upper[1]) * np.random.random_sample()
            w = self.w_in_plane(u, v)

            glass_r = glass_r_range[0] + (glass_r_range[1] - glass_r_range[0]) * np.random.random_sample()
            r_sphere = glass_r / math.sin(self.tau)

            g_c = np.array([u, v, w])
            c = g_c - self.normal * r_sphere * math.cos(self.tau)

            print('uvw:', [u, v, w])
            print('g_c:', g_c)
            print('c:', c)

            self.g_centers.append(g_c)
            self.g_radius.append(glass_r)
            self.centers.append(c)
            self.radius.append(r_sphere)

    def point_dist(self, p1, p2):
        dist = np.sqrt(np.sum((p1 - p2) ** 2))
        return dist

    def check_raindrop_overlapped(self, g_c, max_overlap=0.2):
        for g_c_in in self.g_centers:
            if self.point_dist(g_c, g_c_in) < self.glass_r * 2 * (1-max_overlap):
                return True
        return False

    def get_sphere_raindrop_physics(self, W, H, mode=None, keep_seeds=False, N=None, check_overlap=True):
        self.g_centers = []
        self.g_radius = []
        self.radius = []
        self.centers = []

        # only render raindrops in center FOV 1.5
        left_upper = self.to_glass(W / 6, H / 6)
        left_bottom = self.to_glass(W / 6, 5 * H / 6)
        right_upper = self.to_glass(5 * W / 6, H / 6)
        right_bottom = self.to_glass(5 * W / 6, 5 * H / 6)

        glass_r_range = (0.15 * 10, 0.6 * 10)  # mm
        if keep_seeds:
            glass_r = 5
        else:
            glass_r = glass_r_range[0] + (glass_r_range[1] - glass_r_range[0]) * np.random.random_sample()
        low_glass_r = glass_r - 0.5
        high_glass_r = glass_r + 0.5
        self.glass_r = glass_r

        # use cm to calculate density
        # *1.5 as the raindrops appear outside center crop
        max_density = 1 / (np.pi * (glass_r / 10) ** 2) * 1.5 * 1.5  # scale density to 1.5x
        min_density = 1 / (np.pi * (glass_r / 10) ** 2) * 1.5 * 0.7
        self.density = min_density + \
                       (max_density - min_density) * np.random.random_sample()
        print('min density', min_density, 'max density', max_density)

        if N is None:
            self.n = int(self.visible_area_cm2 / (1.5 ** 2) * self.density)
        else:
            self.n = N

        print('current raindrop num:', self.n, 'raindrop_range:', low_glass_r, high_glass_r, max_density)
        print('scale 1.2x to prevent too many overlapped raindrops')
        if keep_seeds:
            np.random.seed(66)
        for _ in range(int(self.n * 1.2)):
            u = left_bottom[0] + (right_bottom[0] - left_bottom[0]) * np.random.random_sample()
            v = left_upper[1] + (right_bottom[1] - left_upper[1]) * np.random.random_sample()
            w = self.w_in_plane(u, v)

            glass_r = low_glass_r + (high_glass_r - low_glass_r) * np.random.random_sample()
            r_sphere = glass_r / math.sin(self.tau)

            g_c = np.array([u, v, w])
            c = g_c - self.normal * r_sphere * math.cos(self.tau)

            if check_overlap:
                if not self.check_raindrop_overlapped(g_c):
                    self.g_centers.append(g_c)
                    self.g_radius.append(glass_r)
                    self.centers.append(c)
                    self.radius.append(r_sphere)
            else:
                self.g_centers.append(g_c)
                self.g_radius.append(glass_r)
                self.centers.append(c)
                self.radius.append(r_sphere)
        print('current non-overlapped raindrop num:', len(self.g_centers))

    def norm(self, nb_list):
        return LA.norm(nb_list)

    def in_sphere_raindrop(self, x, y):
        p = np.array(self.to_glass(x, y))
        for i in range(len(self.g_centers)):
            if self.norm(p - np.array(self.g_centers[i])) <= self.g_radius[i]:
                return i
        return -1

    def to_sphere_section_env(self, x, y, id):
        center = self.centers[id]
        r_sphere = self.radius[id]

        p_g = np.array(self.to_glass(x, y))
        alpha = math.acos(np.dot(p_g, self.normal) / self.norm(p_g))
        beta = math.asin(self.n_air * math.sin(alpha) / self.n_water)

        po = p_g - self.o_g
        po = po / self.norm(po)
        i_1 = self.normal + math.tan(beta) * po
        i_1 = i_1 / self.norm(i_1)

        oc = p_g - center
        tmp = np.dot(i_1, oc)
        d = -1 * tmp + np.sqrt(tmp ** 2 - np.dot(oc, oc) + r_sphere ** 2)
        p_w = p_g + d * i_1

        normal_w = p_w - center
        normal_w = normal_w / self.norm(normal_w)

        d = (np.dot(p_w, normal_w) - np.dot(normal_w, p_g)) / np.dot(normal_w, normal_w)
        p_a = p_w - (d * normal_w + p_g)
        p_a = p_a / self.norm(p_a)

        eta = math.acos(np.dot(normal_w, p_w - p_g) / self.norm(p_w - p_g))
        # print(eta, self.gamma)
        if eta >= self.gamma:
            assert "total refrection"

        theta = math.asin(self.n_water * math.sin(eta) / self.n_air)
        i_2 = normal_w + math.tan(theta) * p_a

        p_e = p_w + ((self.B - p_w[2]) / i_2[2]) * i_2
        p_i = (self.intrinsic @ p_e.T / self.B).T
        p_i = np.round(p_i)

        return p_i

    # @jit(nopython=True)
    def render(self, rain_image, rain_image_copy, mask, x_range, mode):
        h, w, _ = rain_image.shape
        print('total render pixels:', w * h)

        for x in x_range:
            for y in range(h):
                # find if current pixel's corresponding 3D point on glass is a raindrop point.
                i = self.in_sphere_raindrop(x, y)
                p = None
                if i != -1:
                    try:
                        p = self.to_sphere_section_env(x, y, i)
                    except:
                        rain_image_copy[y, x] = 0
                        mask[y, x] = 1
                        continue
                    u = p[0]
                    v = p[1]
                    if u >= w:
                        u = w - 1
                    elif u < 0:
                        u = 0

                    if v >= h:
                        v = h - 1
                    elif v < 0:
                        v = 0

                    rain_image_copy[y, x] = rain_image[int(v), int(u)]
                    mask[y, x] = 1
        # return rain_image_copy, mask

    def render_multiprocess(self, rain_image, rain_image_copy, mask, x_range, h, w, mode, counter):
        for x in x_range:
            # for x in range(w):
            for y in range(h):
                counter.value += 1
                if counter.value % 50000 == 0:
                    print('rendered pixels:', counter.value)
                i = self.in_sphere_raindrop(x, y)
                # print(i)
                p = None
                if i != -1:
                    try:
                        p = self.to_sphere_section_env(x, y, i)
                    except:
                        rain_image_copy[y * w + x] = 0
                        mask[y * w + x] = 1
                        continue
                    u = p[0]
                    v = p[1]
                    if u >= w:
                        u = w - 1
                    elif u < 0:
                        u = 0

                    if v >= h:
                        v = h - 1
                    elif v < 0:
                        v = 0

                    rain_image_copy[y * w + x] = rain_image[int(v) * w + int(u)]
                    mask[y * w + x] = 1

    def render_multiprocess_LR(self, L_rain_image, L_rain_image_copy, R_rain_image, R_rain_image_copy, mask, x_range, h,
                               w, mode, counter):
        for x in x_range:
            for y in range(h):
                counter.value += 1
                if counter.value % 50000 == 0:
                    print('rendered pixels:', counter.value)
                i = self.in_sphere_raindrop(x, y)
                p = None
                if i != -1:
                    try:
                        p = self.to_sphere_section_env(x, y, i)
                    except:
                        L_rain_image_copy[y * w + x] = 0
                        R_rain_image_copy[y * w + x] = 0
                        mask[y * w + x] = 1
                        continue
                    u = p[0]
                    v = p[1]
                    if u >= w:
                        u = w - 1
                    elif u < 0:
                        u = 0

                    if v >= h:
                        v = h - 1
                    elif v < 0:
                        v = 0

                    L_rain_image_copy[y * w + x] = L_rain_image[int(v) * w + int(u)]
                    R_rain_image_copy[y * w + x] = R_rain_image[int(v) * w + int(u)]
                    mask[y * w + x] = 1
