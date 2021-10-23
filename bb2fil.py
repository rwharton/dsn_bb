import numpy as np
import os 
import glob
import time
import subprocess
import astropy.units as u
from astropy.time import Time
import sigproc as fb
import click

#########################
##  DADA HEADER CLASS  ##
#########################

class DADA_HDR:
    def __init__(self):
        self.obs_id = "unset"
        self.filename = "unset"
        self.filenum = 0
        self.file_size = 0
        self.obs_offset = 0
        self.obs_overlap = 0
        self.source_name = "unset"
        self.telescope = "DSN"
        self.receiver = "unset"
        self.nbit = 8 
        self.ndim = 2 
        self.npol = 1 
        self.nchan = 7
        self.tsamp = 0.0  # microsec
        self.bw = 0.0     # MHz
        self.utc_start = "2021-03-02-23:52:10.999999749"
        self.mjd_start = 0.0
        self.freq = 0.0 

    def ascii_hdr(self):
        outstr =  (
        "HEADER DADA\n" + 
        "HDR_VERSION 1.0\n" + 
        "HDR_SIZE 4096\n" + 
        "DADA_VERSION 1.0\n" +
        "OBS_ID %s\n" %(self.obs_id) + 
        "PRIMARY unset\n" + 
        "SECONDARY unset\n" + 
        "FILE_NAME %s\n" %(self.filename) + 
        "FILE_NUMBER %d\n" %(self.filenum) + 
        "FILE_SIZE %d\n" %(self.file_size) +  
        "OBS_OFFSET %d\n" %(self.obs_offset) + 
        "OBS_OVERLAP %s\n" %(self.obs_overlap) + 
        "SOURCE %s\n" %(self.source_name) + 
        "TELESCOPE %s\n" %(self.telescope) + 
        "INSTRUMENT DADA\n" + 
        "RECEIVER %s\n" %(self.receiver) + 
        "NBIT %d\n" %(self.nbit) + 
        "NDIM %d\n" %(self.ndim) + 
        "NPOL %d\n" %(self.npol) + 
        "NCHAN %d\n" %(self.nchan) + 
        "RESOLUTION 1\n" + 
        "DSB 1\n" + 
        "TSAMP %.8f\n" %(self.tsamp) + 
        "BW %8f\n" %(self.bw) + 
        "UTC_START %s\n" %(self.utc_start) + 
        "MJD_START %.16f\n" %(self.mjd_start) + 
        "FREQ %.8f\n" %(self.freq) + 
        "# end of header\n")
        return outstr

    def write_header(self, outfile):
        """
        Write PSRDADA header -- this consists of 
        an ASCII part then pad "\x00" until you 
        get to 4096 bytes  
        """ 
        # First we need to add the file name to header
        self.filename = outfile

        # Next we get the byte size of the header
        hdr_str = self.ascii_hdr()
        hdr_bytes = len(hdr_str.encode('utf-8'))
        
        # Now get number of bytes to fill to 4096
        nbyte_fill = 4096 - hdr_bytes

        # We will pad the header using unsigned int 0s
        fill_bytes = np.zeros(nbyte_fill, dtype='uint8') 

        # Open the file and write ascii header
        with open(outfile, 'w') as fout:
            fout.write(hdr_str)

        # Now append the file with padded 0s
        with open(outfile, 'ab') as fout:
            fout.write(fill_bytes)

        return


###############################
##  READ DSN CS DATA FILES   ##
###############################

def get_cs_info(infile):
    hd, hsize, err = fb.read_header(infile, 4, fb.fmtdict)
    if err:
        print("error %d" %(err))
        return 0
    else: pass
    return hd

def check_and_sort_files(infiles, reverse=False):
    """
    Check basic consistency of input files 
    
    sort in in ascending order if reverse=False, 
    otherwise in descending order
    """
    hdlist = []
    for infile in infiles:
        hd_ii = get_cs_info(infile)
        hdlist.append(hd_ii)
    
    freqs = np.array([ hd['fch1'] for hd in hdlist ])
    bws = np.array([ hd['foff'] for hd in hdlist ])
    tstarts = np.array([ hd['tstart'] for hd in hdlist ])
    srcs = np.array([ hd['source_name'] for hd in hdlist ])

    # check consistency
    err = 0 
    if len( np.unique(bws) ) > 1:
        print("Multiple BWs:")
        print(np.unique(bws))
        err += 1
    else: pass

    if len( np.unique(srcs) ) > 1:
        print("Multiple source names")
        print(np.unique(srcs))
        err += 1
    else: pass

    if len( np.unique(tstarts) ) > 1:
        print("Multiple MJD start dates")
        print(np.unique(tstarts))
        err += 1
    else: pass

    if err:
        print("Files incompatible")
        return 0
    else: pass

    # Now sort files by frequency
    xx = np.argsort(freqs)
    
    # Flip if required
    if reverse:
        xx = xx[::-1]
    else: pass

    freqsort = np.array([ freqs[xxi] for xxi in xx ])
    filesort = [infiles[xxi] for xxi in xx]

    # Get global pars
    src = srcs[0]
    bw  = bws[0]
    tstart = tstarts[0]

    return filesort, freqsort, src, bw, tstart 
     

def read_cs(infile, start=0, count=-1, hdr_size=None):
    """
    Read complex voltage data (64-bit complex values)

    start = sample number to start at

    count = number of samples to read after start

    hdr_size = size of header in bytes, if None, read 
               from SIGPROC header
    
    codd header is 249, cs is 316
    """
    if hdr_size is None:
        _, hsize, _ = fb.read_header(infile, 4, fb.fmtdict)
        hdr_size = hsize
    else: pass 

    # Offset in bytes
    offset = hdr_size + (start * 8)

    # Open file in read mode
    with open(infile, 'rb') as fin:
        dat = np.fromfile(fin, offset=offset, count=count, dtype='complex64')

    return dat


def read_many_cs(infiles, start=0, count=-1, hdr_size=None):
    """
    Read complex voltage data from many files
    """
    # If count is -1, read one file to get count
    if count < 0:
        dtmp = read_cs(infiles[0], start=start, count=-1, hdr_size=hdr_size)
        count = len(dtmp)
    else: pass

    N = len(infiles)
    dd_all = np.zeros( (N, count), dtype='complex64')

    for ii in range(N):
        fii = infiles[ii]
        dii = read_cs(fii, start=start, count=count, hdr_size=hdr_size)
        dd_all[ii, :] = dii[:]

    # Make big for testing
    #dd_all = np.hstack( [dd_all] * 10 )

    return dd_all


##############################
##  WRITE TO DADA FILE(S)   ##
##############################

def write_to_dada(outfile, data, fcenter_MHz, bw_MHz, tsamp_us, 
                  mjd_start, nchan, source_name=None, fac=10):
    """
    Data in shape (Nt, Nchan)

    Note: we are fixing bits per sample = 8 and npol = 1
    """
    # Fixed values
    bps = 8
    npol = 1

    # Check data shape
    if data.shape[1] == nchan:
        pass
    else:
        print("Data shape mismatch")
        print("%d chans in data instead of %d" %(data.shape[1], nchan))
        return 0

    # Scale data 
    data = data * fac

    # Get data in the right format 
    ddr = np.real(data.ravel()).astype('int8')
    ddi = np.imag(data.ravel()).astype('int8')
    dd_out = np.reshape( np.vstack( (ddr, ddi) ).T, (-1, 2 * nchan) )

    # Now we can start filling out the header
    hdr = DADA_HDR()

    # File size is number of bytes in payload (not header)
    hdr.file_size = dd_out.size
    
    # Source name if available
    if source_name is not None:
        hdr.source_name = source_name

    # Bits per sample
    hdr.nbit = bps

    # Number of channels
    hdr.nchan = nchan

    # Sample time in microseconds
    hdr.tsamp = tsamp_us

    # Set the total bandwidth in MHz
    hdr.bw = bw_MHz

    # UTC Start Time
    tstart = Time(mjd_start, format='mjd')
    tstr = str(tstart.datetime64)
    utc_str = '-'.join(tstr.split('T'))
    hdr.utc_start = utc_str
    
    # MJD start
    hdr.mjd_start = mjd_start
    
    # Center Frequency in MHz
    hdr.freq = fcenter_MHz

    # Now that the header is set, we can write it to file
    hdr.write_header(outfile)

    # Now we can append the data to this file
    with open(outfile, 'ab') as fout:
        fout.write(dd_out)
    
    # and that should do it
    return 


#############################
##  CS -> Dynamic Spectrum  #
#############################

def cs2dada(basename, indir, outdir):
    """
    Get CS baseband files of the form:

        {basename}*.cs

    in "indir" directory, and convert 
    them to a DADA file of the form 

        basename.dada

    in the directory "outdir"
    """
    t0 = time.time()
    # Get files 
    infiles = glob.glob("%s/%s*.cs" %(indir, basename))
   
    # Check files and get info 
    check_out = check_and_sort_files(infiles, reverse=True)  

    if check_out == 0:
        return 0
    else:
        pass

    sfiles, freqs, src, bw, tstart = check_out

    # Read data (shape will be (nchan, nspec))
    dd = read_many_cs(sfiles)

    # Get nec info for DADA 
    nchan = len(freqs)
    full_bw_MHz = -1 * np.abs(bw) * nchan
    fcenter_MHz = np.mean(freqs)
    tsamp_us = 1.0 / np.abs(bw)
    mjd_start = tstart

    # Write data as (nspec, nchan) shape 
    dada_out = "%s/%s.dada" %(outdir, basename)
    
    write_to_dada(dada_out, dd.T, fcenter_MHz, full_bw_MHz, 
                  tsamp_us, mjd_start, nchan, source_name=src) 

    t1 = time.time()
    print("DADA conversion -- %.1f seconds" %(t1-t0))
    return


def run_digifil(infile, outfile, dm, nchan, nthread=1, inc_ddm=True):
    """
    Use digifil to coherently de-disperse and channelize 
    the 7-channel baseband datat in the DADA file

    infile:  DADA file to process

    outfile: name of output *.fil file

    dm:  DM for coherent de-dispersion

    nchan: total number of channels in output filterbank 

    nthreads: Number of threads for proc (default=1)

    inc_ddm: Also do the incoherent de-dispersion? (default=True)
    """
    # Use options to build up command 
    cmd = "digifil"

    # add "-b-32" to output 32-bit floats 
    cmd += " -b-32"

    # add -IO to turn of normalization
    cmd += " -I0"

    # add number of threads
    cmd += " -threads %d" %(nthread)

    # add channelization 
    cmd += " -F %d:D" %(nchan)

    # add dm 
    cmd += " -D %.4f" %(dm)

    # add incoherent flag if required
    if inc_ddm:
        cmd += " -K"
    else: pass

    # add output file name
    cmd += " -o %s" %(outfile)

    # add input file name
    cmd += " %s" %(infile) 

    # Print command
    print(cmd)

    # Run command
    t0 = time.time()
    subprocess.run(cmd, shell=True)
    t1 = time.time()

    # print time
    #print("digifil step -- %.1f seconds" %(t1-t0))
    
    return


def get_chunk_base(basename, cs_dir):
    """
    get the unique {basename}-XXXX values
    """
    # Get the paths
    pfiles = glob.glob("%s/%s*cs" %(cs_dir, basename))
    
    # Get file names 
    fnames = np.array([ pf.split("/")[-1] for pf in pfiles ])

    # Get chunk bases
    cnames = np.array([ fn.rsplit('-', 1)[0] for fn in fnames ])

    # Get sorted list of unique values
    unames = np.unique(cnames)

    return unames   


def cs2fil(basename, cs_dir, dada_dir, fil_dir, dm, nchan, 
           nthread=1, inc_ddm=False):
    """
    Take cs files of form {basename}*.cs in directory cs_dir 
    and convert them into a single DADA file {basename}.dada 
    in dada_dir.

    Then run digifil to produce a coherently de-dispersed 
    filterbank at DM "dm" with a total of nchan channels.
    Use nthread threads for digifil and remove incohrent 
    dispersive delay if inc_ddm=True 
    """
    t0 = time.time()
    # Read in and convert data to DADA file
    cs2dada(basename, cs_dir, dada_dir)
    
    t1 = time.time()

    # Run digifil 
    dada_file = "%s/%s.dada" %(dada_dir, basename)
    fil_file = "%s/%s.fil" %(fil_dir, basename)
    run_digifil(dada_file, fil_file, dm, nchan, nthread=nthread, inc_ddm=inc_ddm)
    t2 = time.time()

    # Clean up by removing dada file
    #if os.path.exists(fil_file) and os.path.exists(dada_file):
    #    os.remove(dada_file)

    print("")
    print("Convert to DADA -- %.1f sec" %(t1 - t0))
    print("digifil         -- %.1f sec" %(t2 - t1))
    print("")
    print(" Total = %.1f sec" %(t2-t0))

    return

@click.command()
@click.option("--basename", type=str, 
              help="Name of cs files: {basename}*.cs")
@click.option("--cs_dir", type=str,
              help="Directory containing cs files")
@click.option("--dada_dir", type=str, 
              help="Directory for DADA files")
@click.option("--fil_dir", type=str,
              help="Directory for *.fil files")
@click.option("--dm", type=float, 
              help="DM for intra-channel coherent dedispersion")
@click.option("--nchan", type=int, 
              help="Total number of output channels in filterbank")
@click.option("--nthread", type=int, default=1,  
              help="Number of threads for digifil processing")
def cs2fil_multi(basename, cs_dir, dada_dir, fil_dir, dm, nchan, 
                 nthread=1, inc_ddm=False):
    """
    Convert multiple chunks of complex sampled voltage data 
    to coherently de-dispersed channelized filterbanks.
    
    Input cs files assumed to be in the form 
 
           {basename}-XXXX-YY.cs in directory 
    
    where {basename} is the base name associated with all 
    files, XXXX is a number 0000-9999 that gives the data 
    chunk order, and YY is 01-07 and represents the channel.

    The channels of each time chunk will be combined into a 
    DADA file called 

           {basename}-XXXX.dada

    Then run digifil to produce a coherently de-dispersed 
    filterbank at DM "dm" with a total of nchan channels.
    Use nthread threads for digifil and remove incohrent 
    dispersive delay if inc_ddm=True 

    The filterbank file will be called 

          {basename}-XXXX.fil
    """
    # First get a list of unique {basename}-XXXX values
    chunk_bases = get_chunk_base(basename, cs_dir)

    # Make sure we actually have some files
    if len(chunk_bases) == 0:
        print("No files found with basename %s in %s" %(basename, cs_dir))
        return 0
    else: pass

    for cbase in chunk_bases:
        print("Processing %s..." %cbase)
        cs2fil(cbase, cs_dir, dada_dir, fil_dir, dm, nchan, 
           nthread=nthread, inc_ddm=inc_ddm)
    
    return
    
if __name__ == "__main__":
    cs2fil_multi()

