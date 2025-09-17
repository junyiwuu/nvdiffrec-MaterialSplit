from pxr import Usd, UsdGeom, Gf
import numpy as np
import os

asset_path = r"/mnt/D/2025/research/customize_dataset/pile/render_selfRotate_outdoor/"

usd_path = os.path.join(asset_path, "scene.usd")
cam_path = "/cam"
turntable_path = "/pile/geom/render/pile"


stage = Usd.Stage.Open(usd_path)
stage.Load()

cam_prim = stage.GetPrimAtPath(cam_path)
turntable_prim = stage.GetPrimAtPath(turntable_path)


cam = UsdGeom.Camera(cam_prim)

imageW = 512.0
imageH = 512.0

# get stage timeline (Houdini scene timeline)
def timeline(stage, turntable_prim):


    start_time = stage.GetStartTimeCode()
    end_time = stage.GetEndTimeCode()
    
    if start_time==end_time:
        return [Usd.TimeCode.Default()]
    
    # return times
    return (Usd.TimeCode(t) for t in range(int(start_time), int(end_time)+1))



def write_poses_bounds(out_path, c2w_list, imageH, imageW, focalLen_pix, near, far):

    N = len(c2w_list)
    poses_bounds = np.zeros((N, 17), dtype=np.float32)

    for i, c2w in enumerate(c2w_list):
        c2w = c2w.astype(np.float32)


        t = c2w[3, :3].astype(np.float32)

        right = c2w[0, :3]
        up    = c2w[1, :3]
        back  = c2w[2, :3]

        pose3x5 = np.zeros((3, 5), dtype=np.float32)
    
        pose3x5[:, 0] = -up
        pose3x5[:, 1] = right       
        pose3x5[:, 2] = back 
        pose3x5[:, 3] =  t  

        pose3x5[0, 4] = float(imageH)
        pose3x5[1, 4] = float(imageW)
        pose3x5[2, 4] = float(focalLen_pix)

        poses_bounds[i, :15] = pose3x5.reshape(-1)
        poses_bounds[i, 15]  = float(near)
        poses_bounds[i, 16]  = float(far)


    np.save(out_path, poses_bounds)


def main():

    times = timeline(stage, turntable_prim)
    c2w_list = []
    focalLen_pix = None

    #load camera
    cam = UsdGeom.Camera(cam_prim)


    # clipping range
    clip = cam.GetClippingRangeAttr().Get()
    near = clip[0]
    far = clip[1]

    #create a cache
    xform_cache = UsdGeom.XformCache()

    for time in times:
        xform_cache.SetTime(time)
        cam2w = xform_cache.GetLocalToWorldTransform(cam_prim)   #4x4 matrix
        o2w = xform_cache.GetLocalToWorldTransform(turntable_prim)   
        o2w_inv = Gf.Matrix4d(o2w).GetInverse()

        c2w = cam2w * o2w_inv
        c2w_np = np.array(c2w, dtype=np.float64)

        focalLength = cam.GetFocalLengthAttr().Get(time)
        horiz_aper_pix = cam.GetHorizontalApertureAttr().Get(time)
        verti_aper_pix = cam.GetVerticalApertureAttr().Get(time)



        if focalLen_pix is None:
            focalLen_pix = float(focalLength) / float(horiz_aper_pix) * float(imageW) 

        c2w_list.append(c2w_np.astype(np.float32))


    out_path = os.path.join("data/pile_cus_turntable/", "poses_bounds.npy")
    write_poses_bounds(out_path, c2w_list, imageH, imageW, focalLen_pix, near, far)



if __name__ == "__main__":
    main()









































