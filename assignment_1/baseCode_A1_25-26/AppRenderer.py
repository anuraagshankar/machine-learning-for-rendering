from PyRT_Core import *
from PyRT_Integrators import *
import time

def sphere_scene(envMap=None):
    # Create a scene object
    scene_ = Scene()
    i_a = RGBColor(0.5, 0.5, 0.5)
    scene_.set_ambient(i_a)

    # Create the materials (BRDF)
    white_diffuse = Lambertian(RGBColor(0.8, 0.8, 0.8))
    green_diffuse = Lambertian(RGBColor(0.2, 0.8, 0.2))

    # Create the Scene Geometry (3D objects)
    # sphere
    radius = 2
    sphere = Sphere(Vector3D(0.0, 0.0, -5.0), radius)
    sphere.set_BRDF(white_diffuse)
    scene_.add_object(sphere)

    # Finite plane
    side = 4 * radius
    half_side = side / 2
    plane_point = Vector3D(-half_side, -radius, -5.0 + half_side)
    right_vector = Vector3D(side, 0.0, 0.0)
    front_vector = Vector3D(0.0, 0.0, -side)
    plane = Parallelogram(plane_point, right_vector, front_vector)
    plane.set_BRDF(green_diffuse)
    scene_.add_object(plane)

    # Create a Point Light Source
    point_light = PointLight(Vector3D(0.0, 5.0, 0.0), RGBColor(80, 80, 80))
    scene_.add_point_light_sources(point_light)

    if envMap is not None:
        # Set up an environment map
        scene_.set_environment_map(envMap)

    # Create the camera
    width = 500
    height = 500
    vertical_fov = 60
    camera = Camera(width, height, vertical_fov)
    scene_.set_camera(camera)

    return scene_

# --------------------------------------------------Set up variables
FILENAME = 'phong'
DIRECTORY = 'out/'
#env_map_path = 'env_maps/black_and_white.hdr'
#env_map_path = 'env_maps/outdoor_umbrellas_4k.hdr'
#env_map_path = 'env_maps/outdoor_umbrellas_4k_clamped.hdr'
env_map_path = 'env_maps/arch_nozero.hdr'

# -------------------------------------------------Main
# Create Integrator
integrator = PhongIntegrator(DIRECTORY + FILENAME)

# Create the scene
scene = sphere_scene(envMap=env_map_path)

# Attach the scene to the integrator
integrator.add_scene(scene)

# Render!
start_time = time.time()
integrator.render()
end_time = time.time() - start_time
print("--- Rendering time: %s seconds ---" % end_time)

# -------------------------------------------------open saved npy image
image_nd_array = np.load(integrator.get_filename() + '.npy')
tonemapper = cv2.createTonemap(gamma=2.5)
image_nd_array_ldr = tonemapper.process(image_nd_array.astype(np.single)) * 255.0
cv2.imshow('PyRT - Python Ray Tracer for MLR', cv2.cvtColor(image_nd_array_ldr.astype(np.uint8), cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
