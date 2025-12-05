import numpy as np
from astropy.coordinates import SkyCoord, BarycentricTrueEcliptic
import pickle
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import matplotlib.colors as colors
from matplotlib.image import imread

def rebin(arr, new_shape):
    """Rebin 2D array arr to shape new_shape by averaging."""
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
            new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

def get_tois(fn_tois):

    tois = pd.read_csv(fn_tois)

    coords = SkyCoord(ra=tois['RA'], dec=tois['Dec'], unit=(u.deg, u.deg))
    ecoords = coords.transform_to(BarycentricTrueEcliptic)
    eclon = ecoords.lon.value * 1
    eclat = ecoords.lat.value * 1

    ecliptic_coords = True

    if ecliptic_coords:
        lon_pl = ecoords.lon.value * 1
        lat_pl = ecoords.lat.value * 1
    else:
        lon_pl = coords.ra.value * 1
        lat_pl = coords.dec.value * 1

    lon_pl -= 180.
    lon_pl *= -1.

    for ii in np.arange(tois.shape[0]):
        if type(tois.loc[ii, 'TFOPWG Disposition']) is not str:
            tois.loc[ii, 'TFOPWG Disposition'] = 'PC'

    cands = (tois['TFOPWG Disposition'] == 'APC') | \
            (tois['TFOPWG Disposition'] == 'PC')
    conf = (tois['TFOPWG Disposition'] == 'CP') | \
        (tois['TFOPWG Disposition'] == 'KP')

    selcn = cands
    selcf = conf

    cansz = 5
    consz = 15

    return lon_pl, lat_pl, cands, conf, selcn, selcf

def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['lon'], data['lat'], data['data'], data['metadata']

def read_pickle_files(base_path, file_names):

    data_sets = [load_pickle(f"{base_path}/{file}") for file in file_names]

    lon_data_all = [item for sublist in [data[0] for data in data_sets] for item in sublist]
    lat_data_all = [item for sublist in [data[1] for data in data_sets] for item in sublist]
    img_data_all = [item for sublist in [data[2] for data in data_sets] for item in sublist]

    return lon_data_all, lat_data_all, img_data_all

def projection_param(hemisphere):

    if hemisphere == 'south':
        cenlon = 90.
        cenlat = -90.#-66.560708333333#
        projection = ccrs.AzimuthalEquidistant(central_longitude=cenlon, central_latitude=cenlat)    
        wrap = False#True#
        title = "NASA TESS's View\nof the Southern\nHemisphere"
        min_lat, max_lat = -90, 0.
    
    elif hemisphere == 'north':
        cenlon = -90.
        cenlat = 90.#-66.560708333333#
        projection = ccrs.AzimuthalEquidistant(central_longitude=cenlon, central_latitude=cenlat)    
        wrap = False#
        title = "NASA TESS's View\nof the Northern\nHemisphere"
        min_lat, max_lat = 0., 90.
    
    elif hemisphere == 'both':
        cenlon = 0.
        cenlat = 0.
        galactic_coords = False
        if galactic_coords:
            cenlon += 180
        projection = ccrs.Mollweide(central_longitude=0)
        # projection = ccrs.PlateCarree(central_longitude=cenlon)
        wrap = True
        title = "NASA TESS's View\nof the Sky"
        min_lat, max_lat = -90, 90.

    return projection, cenlon, min_lat, max_lat, wrap

def load_pickle_files_and_plot():

    base_path = "/Users/vkostov/Documents/GitHub/tess-sky-map/figs/gif_mollweide_ecliptic_blue"
    file_names = [
        "TESS_full_sky_Cycle_6_bin4.pkl",
        "TESS_full_sky_Cycle_7_bin4_new.pkl",
        "TESS_full_sky_Cycle_1_bin4.pkl",
        "TESS_full_sky_Cycle_2_bin4.pkl",
        "TESS_full_sky_Cycle_3_bin4.pkl",
        "TESS_full_sky_Cycle_4_bin4.pkl",
        "TESS_full_sky_Cycle_5_bin4.pkl"
        ]   
    lon_data_all, lat_data_all, img_data_all = read_pickle_files(base_path, file_names)


    hemisphere = 'both'#'north'#'south'#
    projection, cenlon, min_lat, max_lat, wrap = projection_param(hemisphere) 

    vmin = 150
    vmax = 901.
    # cmap = 'gray'

    img = imread('/Users/vkostov/Documents/GitHub/tess-sky-map/bluecmap.png')
    pic = img[:, 0, :]
    cmap = colors.LinearSegmentedColormap.from_list('NASA_blue', pic[::-1, :],N=pic.shape[0])

    fig = plt.figure(figsize=(40, 20), facecolor='black')
    # fig = plt.figure(figsize=(10, 10), facecolor='black')
    ax = plt.axes([0.0, 0.0, 1.0, 1.0], projection=projection)
    ax.patch.set_alpha(0)

    cnorm = colors.LogNorm(vmin=vmin, vmax=vmax)

    nn_start_ = 0
    nn_stop_ = len(lon_data_all)
    binning_factor = 1

    idx = 0

    for lon, lat, sector_data in zip(lon_data_all[nn_start_:nn_stop_], lat_data_all[nn_start_:nn_stop_], img_data_all[nn_start_:nn_stop_]):
        try:
            sector_data = rebin(sector_data, (sector_data.shape[0] // binning_factor, sector_data.shape[1] // binning_factor))

            if (idx > 44) and (idx <= 48): # too much scattered light S91
                sector_data -= 0.5*np.nanmedian(sector_data)

            if (idx > 360) and (idx <= 362): # too much scattered light S92
                sector_data -= 0.20000015*np.nanmedian(sector_data)
            # if (idx > 362) and (idx <= 364): # too much scattered light S92
            #     sector_data -= 0.2*np.nanmedian(sector_data)
            if (idx > 362) and (idx <= 364): # too much scattered light S92
                sector_data -= 0.25*np.nanmedian(sector_data)
            if (idx > 364) and (idx <= 366): # too much scattered light S92
                sector_data -= 0.5*np.nanmedian(sector_data)
            if (idx > 366) and (idx <= 368): # too much scattered light S92
                sector_data -= 0.75*np.nanmedian(sector_data)

            idx += 1
                
            lon = lon[::binning_factor, ::binning_factor]
            lat = lat[::binning_factor, ::binning_factor]
        except:
            carry_on = True

        if hemisphere == 'both':
            lmin = (((cenlon - 178) + 180) % 360) - 180
            lmax = (((cenlon + 178) + 180) % 360) - 180
            wlon = ((lon - cenlon + 180) % 360) - 180
            if wrap and (lon.max() > lmax) and (lon.min() < lmin):
                bad = ((np.abs(wlon[:-1, :-1] - wlon[:-1, 1:]) > 355.) |
                    (np.abs(wlon[:-1, :-1] - wlon[1:, :-1]) > 355.) |
                    (np.abs(wlon[:-1, :-1] - wlon[1:, 1:]) > 355.))
                maskeddata = np.ma.masked_where(bad, sector_data)
                ax.pcolormesh(lon, lat, maskeddata, norm=cnorm, alpha=1,
                            transform=ccrs.PlateCarree(), cmap=cmap)
            else:
                ax.pcolormesh(lon, lat, sector_data, norm=cnorm, alpha=1,
                            transform=ccrs.PlateCarree(), cmap=cmap)
        else:
            if hemisphere == 'south':
                mask = lat[:-1, :-1] <= 0
            elif hemisphere == 'north':
                mask = lat[:-1, :-1] >= 0
            maskeddata = np.ma.masked_where(~mask, sector_data)
            
            ax.pcolormesh(lon, lat, maskeddata, transform=ccrs.PlateCarree(), norm=cnorm, cmap = cmap) 

            min_lon, max_lon, min_lat, max_lat = -180, 180, min_lat, max_lat
            ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())


        del lon, lat, sector_data
        
    # fn_tois ='/Users/vkostov/Documents/GitHub/tess-sky-map/toi_list_090925.csv'
    # lon_pl, lat_pl, cands, conf, selcn, selcf = get_tois(fn_tois)

    # mask_cand = (min_lat <= lat_pl[cands]) &  (lat_pl[cands] <= max_lat)
    # ax.scatter(lon_pl[selcn][mask_cand], lat_pl[selcn][mask_cand], c='#F1A93B', alpha=1, zorder=1,
    #             marker='o', s=10, transform=ccrs.PlateCarree())

    # mask_conf = (min_lat <= lat_pl[conf]) &  (lat_pl[conf] <= max_lat)
    # ax.scatter(lon_pl[selcf][mask_conf], lat_pl[selcf][mask_conf], c='#7BF9FC', alpha=1, zorder=1,
    #             marker='o', s=20, transform=ccrs.PlateCarree())

    # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    # gl.top_labels = False
    # gl.right_labels = False
    # gl.xlines = True
    # gl.ylines = True
    # gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    # gl.ylocator = mticker.FixedLocator([-90, -45, 0, 45, 90])
    # gl.xformatter = LONGITUDE_FORMATTER
    # gl.yformatter = LATITUDE_FORMATTER
    # gl.xlabel_style = {'size': 10, 'color': 'yellow'}
    # gl.ylabel_style = {'size': 10, 'color': 'yellow'}

    # toi_1994 = np.asarray([147.174575, -51.756564])
    # ax.scatter(toi_1994[0], toi_1994[1], c = 'r', alpha=1, zorder=1, marker='*', s=20, transform=ccrs.PlateCarree())
    # 
    # plt.savefig(f'/Users/vkostov/Desktop/TESS_{hemisphere}_bin{binning_factor*4}.png', dpi=300, bbox_inches='tight')
