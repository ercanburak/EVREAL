import torch


def events_to_image_torch(xs, ys, ps, device=None, sensor_size=(180, 240)):
    """
    Method to turn event tensor to image. Allows for bilinear interpolation.
        :param xs: tensor of x coords of events
        :param ys: tensor of y coords of events
        :param ps: tensor of event polarities/weights
        :param device: the device on which the image is. If none, set to events device
        :param sensor_size: the size of the image sensor/output image
    """
    if device is None:
        device = xs.device

    img_size = list(sensor_size)

    img = torch.zeros(img_size).to(device)
    if xs.dtype is not torch.long:
        xs = xs.long().to(device)
    if ys.dtype is not torch.long:
        ys = ys.long().to(device)
    img.index_put_((ys, xs), ps, accumulate=True)
    return img


def events_to_voxel_torch(xs, ys, ts, ps, num_bins, device=None, sensor_size=(180, 240)):
    """
    Turn set of events to a voxel grid tensor, using temporal bilinear interpolation
    Parameters
    ----------
    xs : list of event x coordinates (torch tensor)
    ys : list of event y coordinates (torch tensor)
    ts : list of event timestamps (torch tensor)
    ps : list of event polarities (torch tensor)
    num_bins : number of bins in output voxel grids (int)
    device : device to put voxel grid. If left empty, same device as events
    sensor_size : the size of the event sensor/output voxels
    Returns
    -------
    voxel: voxel of the events between t0 and t1
    """
    if device is None:
        device = xs.device
    assert (len(xs) == len(ys) and len(ys) == len(ts) and len(ts) == len(ps))
    bins = []
    dt = ts[-1] - ts[0]
    t_norm = (ts - ts[0]) / dt * (num_bins - 1)
    zeros = torch.zeros(t_norm.size())
    for bi in range(num_bins):
        bilinear_weights = torch.max(zeros, 1.0 - torch.abs(t_norm - bi))
        weights = ps * bilinear_weights
        vb = events_to_image_torch(xs, ys, weights, device, sensor_size=sensor_size)
        bins.append(vb)
    bins = torch.stack(bins)
    return bins
