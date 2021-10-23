import numpy as np
import struct
import os

# Some useful functions

def get_string(fin, nbytes):
    tmp = fin.read(nbytes)
    if not tmp: return -1, ''
    nchar = struct.unpack('i', tmp)[0]
    if nchar>80 or nchar<1: return nchar, ''
    kk = fin.read(nchar)
    # new
    kk = kk.decode("utf-8")
    nbytes += nchar
    return nbytes, kk
    
def get_val(fin, fmt, nbytes):
    if fmt == 'i':
        tmp = fin.read(4)
        if not tmp: return None
        val = struct.unpack('i', tmp)[0]
    elif fmt == 'd':
        tmp = fin.read(8)
        if not tmp: return None
        val = struct.unpack('d', tmp)[0]
    elif fmt == 'c':
        val = get_string(fin, nbytes)[1]
    else:
        print('Unrecognized format! - %s' %fmt)
        return None
    return val
        
def read_header(fname, nbytes, hd):
    #fin = open(fname, 'r')
    fin = open(fname, 'rb')
    nb, kk = get_string(fin, nbytes)
    hdrsize = 0
    ret_dict = {}
    err = 0
    #print(kk)
    if kk != 'HEADER_START':
        print("Header not in right format!\n")
        err += 1
    while not err:
        nb, kk = get_string(fin, nbytes)
        if kk is None:
            err += 2
            break
        elif kk=='HEADER_END':
            break
        vv = get_val(fin, hd[kk], nbytes)
        if vv is None:
            err += 4
            break
        ret_dict[kk] = vv
    hdrsize = fin.tell()
    fin.close()
    return ret_dict, hdrsize, err

def telescope_name(tel_id):
    """
    Get telescope name from id
    """
    tel_codes = {0 : 'Fake',
                 1 : 'Arecibo',
                 2 : 'Ooty',
                 3 : 'Nancay',
                 4 : 'Parkes',
                 5 : 'Jodrell',
                 6 : 'GBT',
                 7 : 'GMRT',
                 8 : 'Effelsberg'}
    return tel_codes.get(tel_id, '???')

def backend_name(machine_id):
    """
    Get backend name from id
    """
    be_codes = {0 : 'FAKE',
                1 : 'PSPM',
                2 : 'WAPP',
                3 : 'AOFTM',
                4 : 'BPP',
                5 : 'OOTY',
                6 : 'SCAMP',
                7 : 'GMRTFB',
                8 : 'PULSAR2000'}
    return be_codes.get(machine_id, '???')

def data_category(data_type):
    """
    Find out what kind of data this is from id
    """
    dat_codes = {0 : 'raw data',
                 1 : 'filterbank',
                 2 : 'time series',
                 3 : 'pulse profiles',
                 4 : 'amplitude spectrum',
                 5 : 'complex spectrum',
                 6 : 'dedispersed subbands'}
    return dat_codes.get(data_type, '???')

def datsize_info(fpath, hdr_size, tsamp, nbits, nchan, nifs):
    """
    Get some information about the data on file.
    Return
          dat_size: size of data (excluding header) in bytes
          nspec: number of spectra accumulated
          obs_len: time span of data (seconds))
    """
    file_size = os.path.getsize(fpath)
    dat_size = file_size - hdr_size
    nspec = 8 * (dat_size / nbits) / nchan / nifs
    obs_len = nspec * tsamp 
    return dat_size, nspec, obs_len

def fmt_dec(decj):
    """
    Take double 'decj' and convert it to a proper
    Dec string.  

    Ex:
        543443.56999999995 -> '+54:34:43.6'
    """
    dstr = ('%+.3f' %decj).zfill(11)
    dms = "%s%s:%s:%s" %(dstr[0], dstr[1:3], dstr[3:5], dstr[5:])
    return dms

def fmt_ra(raj):
    """
    Take double 'raj' and convert it to a proper RA string.
    Ex:
        143443.56999999995 -> '14:34:43.6'
    """
    rstr = ('%.4f' %raj).zfill(11)
    hms= "%s:%s:%s" %(rstr[0:2], rstr[2:4], rstr[4:])
    return hms

def get_header_dict(fpath, nbytes, keydict):
    hdict, hsize, err = read_header(fpath, nbytes, keydict)
    if err:
        print("Error reading header!")
        return -1
    else: pass
    
    dsize, nspec, obs_len = datsize_info(fpath, hsize, hdict.get('tsamp'),
                                         hdict.get('nbits'), 
                                         hdict.get('nchans'), 
                                         hdict.get('nifs'))
    
    hd = {}

    # Filename
    hd['fname'] = fpath.split('/')[-1]

    # Header size
    hd['hsize'] = hsize

    # Data size
    hd['dsize'] = dsize

    # Data Type
    hd['data_type'] = data_category(hdict.get('data_type'))

    # -centrics
    hd['pulsarcentric'] = hdict.get('pulsarcentric', 0)
    hd['barycentric'] = hdict.get('barycentric', 0)
    hd['topocentric'] = 0 if (hdict.get('pulsarcentric', 0) or\
                                  hdict.get('barycentric', 0)) else 1
    
    # Telescope
    hd['telescope'] = telescope_name(hdict.get('telescope_id'))
    
    # Backend
    hd['backend'] = backend_name(hdict.get('machine_id'))
    
    # Source Name
    hd['src_name'] = hdict.get('source_name')

    # RA / DEC
    src_raj = hdict.get('src_raj')
    hd['src_raj'] = fmt_ra(src_raj) if src_raj else 0
    
    src_dej = hdict.get('src_dej')
    hd['src_dej'] = fmt_dec(src_dej) if src_dej else 0

    # AZ/ZA Start
    hd['az_start'] = hdict.get('az_start', -1.0)
    hd['za_start'] = hdict.get('za_start', -1.0)
    
    # Frequency Channels
    hd['fch1'] = hdict.get('fch1')
    hd['foff'] = hdict.get('foff')
    hd['nchans'] = hdict.get('nchans')
    hd['nbeams'] = hdict.get('nbeams')
    hd['ibeam']  = hdict.get('ibeam')

    # Time samples
    hd['tstart'] = hdict.get('tstart')
    hd['tsamp']  = hdict.get('tsamp')
    hd['nspec']  = nspec
    hd['obs_len'] = obs_len
    hd['nbits'] = hdict.get('nbits')
    
    # IFs
    hd['nifs'] = hdict.get('nifs')
    
    return hd

def hdr_line(description, value):
    return '{0:<32} : {1}\n'.format(description, value)

def nice_header(hd):
    hdr = ''
    
    # File name
    hdr += hdr_line('Data file', hd.get('fname'))

    # Header Size
    hdr += hdr_line('Header size (bytes)', hd.get('hsize'))

    # Data Size
    if hd.get('dsize'):
        hdr += hdr_line('Data size (bytes)', hd.get('dsize'))

    # The -centrics
    if hd.get('pulsarcentric'):
        hdr += hdr_line('Data type', '%s (pulsarcentric)' %hd.get('data_type'))
    elif hd.get('barycentric'):
        hdr += hdr_line('Data type', '%s (barycentric)' %hd.get('data_type'))
    else:
        hdr += hdr_line('Data type', '%s (topocentric)' %hd.get('data_type'))
    
    # Telescope
    hdr += hdr_line('Telescope', hd.get('telescope'))

    # Backend
    hdr += hdr_line('Datataking Machine', hd.get('backend'))

    # Source Name
    if hd.get('src_name'):
        hdr += hdr_line('Source Name', hd.get('src_name'))
    
    # RA/DEC
    if hd.get('src_raj'):
        hdr += hdr_line('Source RA (J2000)', hd.get('src_raj'))
    if hd.get('src_dej'):
        hdr += hdr_line('Source DEC (J2000)', hd.get('src_dej'))

    # AZ/ZA
    az_start = hd.get('az_start', -1.0)
    if az_start != 0.0 and az_start != -1.0:
        hdr += hdr_line('Start AZ (deg)', az_start)
    
    za_start = hd.get('za_start', -1.0)
    if za_start != 0.0 and za_start != -1.0:
        hdr += hdr_line('Start ZA (deg)', za_start)

    # Frequency Channels
    hdr += hdr_line('Frequency of channel 1 (MHz)', '%.6f' %hd.get('fch1'))
    hdr += hdr_line('Channel bandwidth      (MHz)', '%.6f' %hd.get('foff'))
    hdr += hdr_line('Number of channels', hd.get('nchans'))
    hdr += hdr_line('Number of beams', hd.get('nbeams'))
    hdr += hdr_line('Beam number', hd.get('ibeam'))
    
    # Time Samples
    hdr += hdr_line('Time stamp of first sample (MJD)', '%.12f' %hd.get('tstart'))
    hdr += hdr_line('Sample time (us)', '%.5f' %(hd.get('tsamp', -1) * 1.0e6))
    if hd.get('nspec'):
        hdr += hdr_line('Number of samples', hd.get('nspec'))
    if hd.get('obs_len'):
        hdr += hdr_line('Observation length (minutes)', '%.2f' %hd.get('obs_len'))
    hdr += hdr_line('Number of bits per sample', hd.get('nbits'))
    
    # IFs
    hdr += hdr_line('Number of IFs', hd.get('nifs'))

    return hdr
    

fmtdict = {'telescope_id':'i', 'machine_id':'i', 'data_type':'i',
           'rawdatafile':'c', 'source_name':'c', 'barycentric':'i',
           'pulsarcentric':'i', 'az_start':'d', 'za_start':'d',
           'src_raj':'d', 'src_dej':'d', 'tstart':'d', 'tsamp':'d',
           'nbits':'i', 'nsamples':'i', 'fch1':'d', 'foff':'d',
           'nchans':'i', 'nifs':'i', 'refdm':'d', 'period':'d',
           'nbeams':'i', 'ibeam':'i'}
nbytes = 4
