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
    if dt.item() < 1e-9:
        t_norm = torch.linspace(0, num_bins - 1, steps=len(ts))
    else:
        t_norm = (ts - ts[0]) / dt * (num_bins - 1)
    zeros = torch.zeros(t_norm.size())
    for bi in range(num_bins):
        bilinear_weights = torch.max(zeros, 1.0 - torch.abs(t_norm - bi))
        weights = ps * bilinear_weights
        vb = events_to_image_torch(xs, ys, weights, device, sensor_size=sensor_size)
        bins.append(vb)
    bins = torch.stack(bins)
    return bins


def events_to_voxel_grid_pytorch(xs, ys, ts, ps, num_bins, width, height):#, divide_sign=True):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :divide_sign: True, neg/pos divide into two groups
    :param width, height: dimensions of the voxel grid
    :param device: device to use to perform computations
    :return voxel_grid: PyTorch event tensor (on the device specified)
    """


    # assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    with torch.no_grad():
    #events_torch = torch.from_numpy(events)
    #events_torch = events_torch.to(device)
        voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32, device=xs.device).flatten()
        
        if len(ts) == 0:
            return voxel_grid.view(num_bins, height, width)
        # normalize the event timestamps so that they lie between 0 and num_bins

        deltaT = ts[-1] - ts[0]

        if deltaT == 0:
            deltaT = 1.0
        

        # events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
        ts = (num_bins - 1) * (ts - ts[0]) / deltaT
        xs = xs.long()
        ys = ys.long()
        # ts = events[:, 0]
        # xs = events[:, 1].long()
        # ys = events[:, 2].long()
        # pols = events[:, 3].float()
        # pols[pols == 0] = -1  # polarity should be +1 / -1
        ps[ps==0] == -1
        tis = torch.floor(ts)
        tis_long = tis.long()
        dts = ts - tis
        vals_left = ps * (1.0 - dts.float())
        vals_right = ps * dts.float()

        valid_indices = tis < num_bins
        valid_indices &= tis >= 0
        voxel_grid.index_add_(dim=0,
                            index=xs[valid_indices] + ys[valid_indices]
                            * width + tis_long[valid_indices] * width * height,
                            source=vals_left[valid_indices])

        valid_indices = (tis + 1) < num_bins
        valid_indices &= tis >= 0

        voxel_grid.index_add_(dim=0,
                            index=xs[valid_indices] + ys[valid_indices] * width
                            + (tis_long[valid_indices] + 1) * width * height,
                            source=vals_right[valid_indices])


    voxel_grid = voxel_grid.view(num_bins, height, width)

    return voxel_grid
