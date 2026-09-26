"""
Module with methods for extracting compressed file formats.
"""
import os
import zipfile
import tarfile
import gzip
import shutil
import bz2
import hashlib
import requests
import bef4llm.resource_controller.path_helper as ph

ROOT_DIR = os.path.split(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))[0]


def find_extractor(filename):
    """
    Returns suitable extraction method based on the extension of the filename

    Parameters
    ----------
    filename : str
        Name of the file with extension

    Returns
    -------
    Method
    """

    for key in COMPRESSION_FORMATS:
        if filename.endswith(key):
            return lambda dest_dir: COMPRESSION_FORMATS[key](filename, dest_dir)

    raise NotImplementedError(f"File cannot be decompressed. Supported formats are {COMPRESSION_FORMATS.keys()} ")


def zip_extractor(zip_file_path, dest_dir):
    """
    Extracts .zip files

    Parameters
    ----------
    zip_file : str
        filename
    dest_dir : str
        name of a folder

    Returns
    -------
    None

    """

    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
            zip_file.extractall(dest_dir)
            zip_file.close()

        os.renames(zip_file_path, ph.hide_file(zip_file_path))

    except zipfile.BadZipFile as e:
        print(e)
        raise TypeError


def gz_extractor(gz_file_path, dest_dir):
    """
    Extracts .gz and .gzip

    Parameters
    ----------
    gz_file : str
        filename
    dest_dir : str
        name of a folder within rmm4pycore/data

    Returns
    -------
    None
    """

    suffix = gz_file_path.replace("gzip", "gz")
    suffix = suffix.replace(".gz", "")
    suffix = suffix.split(os.sep)[-1]
    # in case a hidden directory is extracted, make the extracted file visible again.
    if suffix.startswith("."):
        suffix = suffix.replace(".", "", 1)
    _dest_file = os.path.join(dest_dir, suffix)

    try:
        os.mkdir(dest_dir)
    except FileExistsError:
        pass

    with gzip.open(gz_file_path, "rb") as f_in, open(_dest_file, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, 65536)

    shutil.move(gz_file_path, ph.hide_file(gz_file_path))


def tar_extractor(tar_file_path, dest_dir):
    """
    Extracts .tar.gz and .tar files

    Parameters
    ----------
    tar_file : str
        filename
    dest_dir : str
        name of a folder within rmm4pycore/data

    Returns
    -------
    None

    """

    if tar_file_path.endswith("tar.gz") or tar_file_path.endswith("tar") or tar_file_path.endswith("tgz"):
        tar = tarfile.open(tar_file_path, "r:*")
        tar.extractall(dest_dir)
        tar.close()

    else:
        raise NotImplementedError("Extraction of supposedly tarfile failed")

    shutil.move(tar_file_path, ph.hide_file(tar_file_path))


def bz2_extractor(bz2_file_path, dest_dir):
    """
    Extracts .bz2

    Parameters
    ----------
    bz2_file : str
        filename
    dest_dir : str
        name of a folder within rmm4pycore/data

    References
    ----------
    [1] https://stackoverflow.com/questions/16963352/decompress-bz2-files

    Returns
    -------
    None
    """

    with open(bz2_file_path, 'rb') as f_in, open(dest_dir, 'wb') as f_out:
        decompressor = bz2.BZ2Decompressor()
        for data in iter(lambda: f_in.read(100 * 1024), b''):
            f_out.write(decompressor.decompress(data))

    shutil.move(bz2_file_path, ph.hide_file(bz2_file_path))


def update_checksum_file(path, checksum):
    """
    Adds a checksum for a new file to .checksums.txt
    Also overwrites existing ones.

    Parameters
    ----------
    path : str
    checksum : str

    Returns
    -------
    None
    """
    checksum_file = ".checksums.txt"
    new_checksum = True
    with open(ph.look_for_file(checksum_file)) as sums:
        with open("temp.txt", "w") as temp:
            for line in sums:
                if not line.startswith(path):
                    temp.write(line)
                else:
                    temp.write(path + " " + checksum + " " + "\n")
                    new_checksum = False
            if new_checksum:
                temp.write(path + " " + checksum + " " + "\n")

    os.remove(ph.look_for_file(checksum_file))
    shutil.move("temp.txt", checksum_file)


def md_5_checksum(filepath):
    """
    Determines md5 checksum of a local file.

    Parameters
    ----------
    filepath

    Returns
    -------
    str

    References
    ----------
    [1] https://stackoverflow.com/questions/54483868/python-compare-local-and-remote-file-md5-hash

    """
    with open(filepath, 'rb') as file:
        m = hashlib.md5()
        while True:
            data = file.read(8192)
            if not data:
                break
            m.update(data)
        checksum = m.hexdigest()
    return checksum


def md_5_remote_checksum(url):
    """
    Determines md5 checksum of a remote file.

    Parameters
    ----------
    url : url to desired file or folder

    Returns
    -------
    str
        checksum
    """
    m = hashlib.md5()
    remote = requests.get(url)
    for data in remote.iter_content(8192):
        m.update(data)
    checksum_remote = m.hexdigest()
    return checksum_remote


COMPRESSION_FORMATS = {
    "zip": zip_extractor,
    "tar": tar_extractor,
    "tar.gz": tar_extractor,
    "raw.tgz": tar_extractor,
    "gz": gz_extractor,
    "gzip": gz_extractor
}

