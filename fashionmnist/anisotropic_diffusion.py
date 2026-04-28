import numpy as numpy
from skimage.transform import downscale_local_mean, resize


def anisotropic_diffusion_(
    img, niter=10, kappa=10, gamma=0.2, voxelspacing=None, option=3, gradient_scale=8
):
    """[summary]

    Args:
        img ([type]): [description]
        niter (int, optional): [description]. Defaults to 1.
        kappa (int, optional): [description]. Defaults to 100.
        gamma (float, optional): [description]. Defaults to 0.2.
        voxelspacing ([type], optional): [description]. Defaults to None.
        option (int, optional): [description]. Defaults to 1.
        gradient_scale (int, optional): [description]. Defaults to 8.

    Returns:
        [type]: [description]
    """

    # define conduction gradients functions
    if option == 1:

        def conduction_gradient(delta, spacing):
            return numpy.exp(-((delta / kappa) ** 2.0)) / float(spacing)

    elif option == 2:

        def conduction_gradient(delta, spacing):
            return 1.0 / (1.0 + (delta / kappa) ** 2.0) / float(spacing)

    elif option == 3:
        kappa_s = kappa * (2 ** 0.5)

        def conduction_gradient(delta, spacing):
            top = 0.5 * ((1.0 - (delta / kappa_s) ** 2.0) ** 2.0) / float(spacing)
            return numpy.where(numpy.abs(delta) <= kappa_s, top, 0)

    # initialize output array
    out = numpy.array(img, dtype=numpy.float32, copy=True)

    # set default voxel spacing if not supplied
    if voxelspacing is None:
        voxelspacing = tuple([1.0] * img.ndim)

    # initialize some internal variables
    deltas = [numpy.zeros_like(out) for _ in range(out.ndim)]

    for _ in range(niter):

        out_sub = downscale_local_mean(out, (gradient_scale, gradient_scale))

        # calculate the diffs
        for i in range(out.ndim):
            delta_sub = numpy.gradient(out_sub, axis=i)
            deltas[i] = resize(delta_sub, out.shape, order=3, anti_aliasing=True)

        # update matrices
        matrices = [
            conduction_gradient(delta, spacing) * delta
            for delta, spacing in zip(deltas, voxelspacing)
        ]

        # update the image
        out += gamma * (numpy.sum(matrices, axis=0))

    return out
