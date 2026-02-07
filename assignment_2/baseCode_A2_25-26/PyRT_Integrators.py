from PyRT_Common import *
from random import randint


# -------------------------------------------------
# Integrator Classes
# -------------------------------------------------
# The integrators also act like a scene class in that-
# it stores all the primitives that are to be ray traced.
# -------------------------------------------------
class Integrator(ABC):
    # Initializer - creates object list
    def __init__(self, filename_, experiment_name=''):
        # self.primitives = []
        self.filename = filename_ + experiment_name
        # self.env_map = None  # not initialized
        self.scene = None

    @abstractmethod
    def compute_color(self, ray):
        pass

    # def add_environment_map(self, env_map_path):
    #    self.env_map = EnvironmentMap(env_map_path)
    def add_scene(self, scene):
        self.scene = scene

    def get_filename(self):
        return self.filename

    # Simple render loop: launches 1 ray per pixel
    def render(self):
        # YOU MUST CHANGE THIS METHOD IN ASSIGNMENTS 1.1 and 1.2:
        cam = self.scene.camera  # camera object
        # ray = Ray()
        print('Rendering Image: ' + self.get_filename())
        for x in range(0, cam.width):
            for y in range(0, cam.height):
                direction = cam.get_direction(x, y)
                ray = Ray(direction=direction)
                pixel = self.compute_color(ray)
                self.scene.set_pixel(pixel, x, y)  # save pixel to pixel array
            progress = (x / cam.width) * 100
            print('\r\tProgress: ' + str(progress) + '%', end='')
        # save image to file
        print('\r\tProgress: 100% \n\t', end='')
        full_filename = self.get_filename()
        self.scene.save_image(full_filename)


class LazyIntegrator(Integrator):
    def __init__(self, filename_):
        super().__init__(filename_ + '_Lazy')

    def compute_color(self, ray):
        return BLACK


class IntersectionIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Intersection')

    def compute_color(self, ray):
        # ASSIGNMENT 1.2: PUT YOUR CODE HERE
        return RED if self.scene.any_hit(ray) else BLACK


class DepthIntegrator(Integrator):

    def __init__(self, filename_, max_depth_=10):
        super().__init__(filename_ + '_Depth')
        self.max_depth = max_depth_

    def compute_color(self, ray):
        # ASSIGNMENT 1.3: PUT YOUR CODE HERE
        hit_data = self.scene.closest_hit(ray)
        
        if not hit_data.has_hit:
            return BLACK

        depth_color = 1 - (hit_data.hit_distance / self.max_depth)
        depth_color = max(0, depth_color)
        return RGBColor(depth_color, depth_color, depth_color)


class NormalIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Normal')

    def compute_color(self, ray):
        # ASSIGNMENT 1.3: PUT YOUR CODE HERE
        hit_data = self.scene.closest_hit(ray)

        if not hit_data.has_hit:
            return BLACK

        normal = hit_data.normal
        normal = Normalize(normal) # not necessary, but to be safe
        color_components = (normal + ONE) / 2
        return RGBColor(color_components.x, color_components.y, color_components.z)


class PhongIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Phong')

    def compute_color(self, ray):
        # ASSIGNMENT 1.4: PUT YOUR CODE HERE
        hit_data = self.scene.closest_hit(ray)

        if not hit_data.has_hit:
            return BLACK

        hit_object = self.scene.object_list[hit_data.primitive_index]

        # Coefficients
        kd = hit_object.get_BRDF().kd
        ks = kd  # Keeping it the same for now
        shininess = 5

        # Unit vectors for Phong lighting equations
        normal = Normalize(hit_data.normal)
        w_o = Normalize(ray.d * -1.0)

        # Ambient light = kd * i_a
        ambient_light = kd.multiply(self.scene.i_a)
        accumulated_color = ambient_light

        # Loop through all light sources
        for light_source in self.scene.pointLights:
            # Light source details
            light_vec = light_source.pos - hit_data.hit_point
            dist_from_light = Length(light_vec)
            incident_intensity = light_source.intensity / dist_from_light**2
            w_i = Normalize(light_vec)

            # Shadow check
            shadow_ray = Ray(hit_data.hit_point, w_i, dist_from_light)
            if self.scene.any_hit(shadow_ray):
                continue

            # Diffuse light = kd * I / d^2 * max(0, n.l)
            diffuse_light = kd.multiply(incident_intensity) * max(0, Dot(normal, w_i))

            # Specular light = ks * I / d^2 * max(0, r.w_o)^s
            two_n_dot_l = 2.0 * Dot(normal, w_i)
            r = Normalize(normal * two_n_dot_l - w_i)  # r = 2 * n.l * n - w_i
            specular_light = ks.multiply(incident_intensity) * max(0, Dot(w_o, r)) ** shininess

            accumulated_color = accumulated_color + diffuse_light + specular_light

        return accumulated_color


class CMCIntegrator(Integrator):  # Classic Monte Carlo Integrator

    def __init__(self, n, filename_, experiment_name=''):
        filename_mc = filename_ + '_MC_' + str(n) + '_samples' + experiment_name
        super().__init__(filename_mc)
        self.n_samples = n

    def compute_color(self, ray):
        pass


class BayesianMonteCarloIntegrator(Integrator):
    def __init__(self, n, myGP, filename_, experiment_name=''):
        filename_bmc = filename_ + '_BMC_' + str(n) + '_samples' + experiment_name
        super().__init__(filename_bmc)
        self.n_samples = n
        self.myGP = myGP

    def compute_color(self, ray):
        pass
