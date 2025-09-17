from pxr import Usd, UsdGeom, Gf
import numpy as np
import os

asset_path = r"/mnt/D/2025/research/customize_dataset/pile/camera/"

usd_path = os.path.join(asset_path, "camera.usd")
cam_path = "/camera"

stage = Usd.Stage.Open(usd_path)
stage.Load()

cam_prim = stage.GetPrimAtPath(cam_path)  

cam = UsdGeom.Camera(cam_prim)

imageW = 1024.0
imageH = 1024.0


def timeline(stage, cam_prim):
    times = set()

    # camera's xform
    xformable = UsdGeom.Xformable(cam_prim)  
   
    for op in xformable.GetOrderedXformOps():
        attr = op.GetAttr()
        times.update(attr.GetTimeSamples())  # times store 0-100 frames (what i have in usd)
    
    # return times
    return (Usd.TimeCode(t) for t in sorted(times))



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

    times = timeline(stage, cam_prim)
    c2w_list = []
    focalLen_pix = None

    #load camera
    cam = UsdGeom.Camera(cam_prim)




    #create a cache
    xform_cache = UsdGeom.XformCache()

    for time in times:
        xform_cache.SetTime(time)
        c2w = xform_cache.GetLocalToWorldTransform(cam_prim)   #4x4 matrix
        c2w_np = np.array(c2w, dtype=np.float64)

        focalLength = cam.GetFocalLengthAttr().Get(time)
        horiz_aper_pix = cam.GetHorizontalApertureAttr().Get(time)
        verti_aper_pix = cam.GetVerticalApertureAttr().Get(time)

        # clipping range
        clip = cam.GetClippingRangeAttr().Get(time)
        near = clip[0]
        far = clip[1]

        if focalLen_pix is None:
            focalLen_pix = float(focalLength) / float(horiz_aper_pix) * float(imageW) 

        c2w_list.append(c2w_np.astype(np.float32))


    out_path = os.path.join("data/customized/pile/", "poses_bounds.npy")
    write_poses_bounds(out_path, c2w_list, imageH, imageW, focalLen_pix, near, far)



if __name__ == "__main__":
    main()









































