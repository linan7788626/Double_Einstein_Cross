import numpy as np
import pylab as pl
import pyfits
#import scipy.ndimage.filters as snf
import scipy.signal as ss
import triangle_root_finding as trf
import alens_arr as aa
import congrid
import scipy.interpolate as sil
from scipy.ndimage.filters import gaussian_filter

#--------------------------------------------------------------------
def make_r_coor(nc,dsx):

    bsz = nc*dsx
    x1 = np.linspace(0,bsz-dsx,nc)-bsz/2.0+dsx/2.0
    x2 = np.linspace(0,bsz-dsx,nc)-bsz/2.0+dsx/2.0

    x2,x1 = np.meshgrid(x1,x2)
    return x1,x2
def make_c_coor(nc,dsx):

    bsz = nc*dsx
    x1,x2 = np.mgrid[0:(bsz-dsx):nc*1j,0:(bsz-dsx):nc*1j]-bsz/2.0+dsx/2.0
    return x1,x2

#--------------------------------------------------------------------
def lens_equation_sie(x1,x2,lpar):
    xc1 = lpar[0]   #x coordinate of the center of lens (in units of Einstein radius).
    xc2 = lpar[1]   #y coordinate of the center of lens (in units of Einstein radius).
    q   = lpar[2]   #Ellipticity of lens.
    rc  = lpar[3]   #Core size of lens (in units of Einstein radius).
    re  = lpar[4]   #Einstein radius of lens.
    pha = lpar[5]   #Orintation of lens.

    phirad = np.deg2rad(pha)
    cosa = np.cos(phirad)
    sina = np.sin(phirad)

    xt1 = (x1-xc1)*cosa+(x2-xc2)*sina
    xt2 = (x2-xc2)*cosa-(x1-xc1)*sina

    phi = np.sqrt(xt2*xt2+xt1*q*xt1*q+rc*rc)
    sq = np.sqrt(1.0-q*q)
    pd1 = phi+rc/q
    pd2 = phi+rc*q
    fx1 = sq*xt1/pd1
    fx2 = sq*xt2/pd2
    qs = np.sqrt(q)

    a1 = qs/sq*np.arctan(fx1)
    a2 = qs/sq*np.arctanh(fx2)

    xt11 = cosa
    xt22 = cosa
    xt12 = sina
    xt21 =-sina

    fx11 = xt11/pd1-xt1*(xt1*q*q*xt11+xt2*xt21)/(phi*pd1*pd1)
    fx22 = xt22/pd2-xt2*(xt1*q*q*xt12+xt2*xt22)/(phi*pd2*pd2)
    fx12 = xt12/pd1-xt1*(xt1*q*q*xt12+xt2*xt22)/(phi*pd1*pd1)
    fx21 = xt21/pd2-xt2*(xt1*q*q*xt11+xt2*xt21)/(phi*pd2*pd2)

    a11 = qs/(1.0+fx1*fx1)*fx11
    a22 = qs/(1.0-fx2*fx2)*fx22
    a12 = qs/(1.0+fx1*fx1)*fx12
    a21 = qs/(1.0-fx2*fx2)*fx21

    rea11 = (a11*cosa-a21*sina)*re
    rea22 = (a22*cosa+a12*sina)*re
    rea12 = (a12*cosa-a22*sina)*re
    rea21 = (a21*cosa+a11*sina)*re

    y11 = 1.0-rea11
    y22 = 1.0-rea22
    y12 = 0.0-rea12
    y21 = 0.0-rea21

    jacobian = y11*y22-y12*y21
    mu = 1.0/jacobian

    res1 = (a1*cosa-a2*sina)*re
    res2 = (a2*cosa+a1*sina)*re
    return res1,res2,mu
#--------------------------------------------------------------------
def xy_rotate(x, y, xcen, ycen, phi):
    phirad = np.deg2rad(phi)
    xnew = (x-xcen)*np.cos(phirad)+(y-ycen)*np.sin(phirad)
    ynew = (y-ycen)*np.cos(phirad)-(x-xcen)*np.sin(phirad)
    return (xnew,ynew)

def gauss_2d(x, y, par):
    #[I0, Re, xc1,xc2,q,pha]
    (xnew,ynew) = xy_rotate(x, y, par[2], par[3], par[5])
    res0 = np.sqrt(((xnew**2)*par[4]+(ynew**2)/par[4]))/np.abs(par[1])
    res = par[0]*np.exp(-7.67*(res0**0.25-1.0))
    return res
#def re_sv(sv,z1,z2):
    #res = 4.0*np.pi*(sv**2.0/vc**2.0)*Da2(z1,z2)/Da(z2)*apr
    #return res

def de_vaucouleurs_2d(x,y,par):
    (xnew,ynew) = xy_rotate(x, y, par[2], par[3], par[5])
    res0 = ((xnew**2)*par[4]+(ynew**2)/par[4])
    res = par[0]*np.exp(-par[1]*res0**0.25)
    return res

#--------------------------------------------------------------------
def main():
    zl = 0.17
    zs = 3.0
    sigmav = 600           #km/s
    re0 = aa.re_sv(sigmav,zl,zs)
    me0 = aa.m200_sv(sigmav,zl)


    #zs_arr = np.linspace(zl,10.0,100)
    #re_arr = zs_arr*0.0
    #for i in xrange(len(zs_arr)):
        #re_arr[i] = aa.re_sv(sigmav,zl,zs_arr[i])

    #pl.figure()
    #pl.plot(zs_arr,re_arr,'-')



    nrbins = 5
    nnn = 128*nrbins
    #dsx = boxsize/nnn
    dsx = 0.396/nrbins # arcsec
    bsz = dsx*nnn # in the units of Einstein Radius
    print re0,me0,bsz

    #nnn = 512
    #bsz = 40 # in the units of Einstein Radius
    #dsx = bsz/nnn # arcsec

    xx01 = np.linspace(-bsz/2.0,bsz/2.0,nnn)+0.5*dsx
    xx02 = np.linspace(-bsz/2.0,bsz/2.0,nnn)+0.5*dsx
    xi2,xi1 = np.meshgrid(xx01,xx02)
    #----------------------------------------------------------------------
    xc1 = 0.0       #x coordinate of the center of lens (in units of Einstein radius).
    xc2 = 0.0       #y coordinate of the center of lens (in units of Einstein radius).
    q   = 0.7       #Ellipticity of lens.
    rc  = 0.0       #Core size of lens (in units of Einstein radius).
    re  = re0       #Einstein radius of lens.
    pha = 5.0      #Orintation of lens.
    lpar = np.asarray([xc1,xc2,q,rc,re,pha])
    #----------------------------------------------------------------------
    ai1,ai2,mua = lens_equation_sie(xi1,xi2,lpar)
    yi1 = xi1-ai1
    yi2 = xi2-ai2

    #pl.figure()
    #pl.contour(xi2,xi1,mua)
    #pl.colorbar()

    #pl.figure()
    #pl.contour(yi2,yi1,mua)
    #pl.colorbar()

    mags_of_sources = 100.0
    ys1 = 0.85
    ys2 = -0.22

    xroot1,xroot2,nroots = trf.roots_zeros(xi1,xi2,ai1,ai2,ys1,ys2)

    idr1 = ((np.array(xroot1)+bsz/2.0-dsx/2.0)/dsx).astype('int')
    idr2 = ((np.array(xroot2)+bsz/2.0-dsx/2.0)/dsx).astype('int')
    g_limage = xi1*0.0
    g_limage[idr1,idr2] = mags_of_sources*np.abs(mua[idr1,idr2])

    ys1 = 0.67
    ys2 = 0.75

    xroot1,xroot2,nroots = trf.roots_zeros(xi1,xi2,ai1,ai2,ys1,ys2)

    idr1 = ((np.array(xroot1)+bsz/2.0-dsx/2.0)/dsx).astype('int')
    idr2 = ((np.array(xroot2)+bsz/2.0-dsx/2.0)/dsx).astype('int')
    g_limage[idr1,idr2] = g_limage[idr1,idr2] + mags_of_sources*np.abs(mua[idr1,idr2])

    #pl.figure()
    #pl.contourf(g_limage)
    #pl.colorbar()

    #pl.figure()
    #pl.contour(yi1,yi2,mua)
    #pl.colorbar()
    #print xroot1,xroot2
    #pl.figure(figsize=(10,10))
    #pl.xlim(-2.0,2.0)
    #pl.ylim(-2.0,2.0)
    #pl.plot(xroot1,xroot2,'go')
    #pl.plot(ys1,ys2,'ko')
    #----------------------------------------------------------------------
    g_amp = 0.1*re      # peak brightness value
    g_sig = 300.*re      # Gaussian "sigma" (i.e., size)
    g_xcen = xc1    # x position of center (also try (0.0,0.14)
    g_ycen = xc2    # y position of center
    g_axrat = q     # minor-to-major axis ratio
    g_pa = pha+90      # major-axis position angle (degrees) c.c.w. from x axis
    gpar = np.asarray([g_amp,g_sig,g_xcen,g_ycen,g_axrat,g_pa])
    #----------------------------------------------------------------------
    #g_simage = gauss_2d(xi1,xi2,gpar) # modeling source as 2d Gaussian with input parameters.
    g_lensimage = gauss_2d(xi1,xi2,gpar)
    g_limage = g_limage + g_lensimage/3.0/nrbins**2.0

    #pl.figure()
    #pl.contourf(g_lensimage)
    #pl.colorbar()

    #file_psf = "./sdsspsf.fits"
    #g_psf = pyfits.getdata(file_psf)-1000.0
    #g_psf = g_psf/np.sum(g_psf)

    #g_psf = congrid.congrid(g_psf,[np.shape(g_psf)[0]*nrbins,np.shape(g_psf)[1]*nrbins])
    #g_psf = sil.RectBivariateSpline(g_psf,
                                #bbox=[np.shape(g_psf)[0]*nrbins,np.shape(g_psf)[1]*nrbins,np.shape(g_psf)[0]*nrbins,np.shape(g_psf)[1]*nrbins], kx=3, ky=3, s=0)

    #print np.shape(g_psf),np.max(g_lensimage)

    #pl.figure()
    #pl.contourf(g_limage)
    #pl.colorbar()


    #file_noise = "./sdssgal.fits"
    #g_noise = pyfits.getdata(file_noise)-1000.0
    #g_limage = ss.fftconvolve(g_limage,g_psf,mode="same")

    #file_noise = "./sdssgal.fits"
    #g_noise = pyfits.getdata(file_noise)-1000.0
    nstd = 0.5001
    g_noise=nstd*np.random.normal(0.0,1.0,(nnn,nnn))
    g_limage = gaussian_filter(g_limage,5)

    ##pl.figure()
    ##pl.contourf(g_limage)
    #pl.colorbar()

    g_limage = g_limage+g_noise

    output_filename = "./test.fits"
    pyfits.writeto(output_filename,g_limage,clobber=True)


    return 0
#------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
    pl.show()
