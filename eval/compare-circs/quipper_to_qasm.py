import os
import sys
import qiskit.qasm3
import os.path
import shutil
import tempfile
import subprocess

QUIPPER_QASM_DIR = os.environ.get("QUIPPER_QASM_DIR")
# Useful for security-minded individuals who heed the warning at the top of
# this page: https://wiki.debian.org/Docker. The default is for the more
# typical (and more insecure) configuration.
DOCKER_USE_SUDO = bool(os.environ.get('DOCKER_USE_SUDO'))
DOCKER_TMPDIR = os.environ.get("DOCKER_TMPDIR")

def run_docker(hs_filename, prog_args=None, mute=False, use_sudo=DOCKER_USE_SUDO):
    with tempfile.TemporaryDirectory(dir=DOCKER_TMPDIR) as tmpdir:
        basename = os.path.basename(hs_filename)
        qasm_basename = basename.rsplit('.', maxsplit=1)[0] + '.qasm'
        qasm_path = os.path.join(tmpdir, qasm_basename)

        shutil.copy(hs_filename, tmpdir)
        args = []
        if use_sudo:
            args += ['sudo', '-g', 'docker']
        args += ['docker', 'run', '--rm', '-v', tmpdir + ':/quipper',
                 'qwerty-arifact-quipper', 'qasm.sh', basename]
        if prog_args:
            args += prog_args
        io_fp = subprocess.DEVNULL if mute else None
        subprocess.run(args, check=True, stdin=io_fp,
                       stdout=io_fp, stderr=io_fp)
        with open(qasm_path) as fp:
            return fp.read()

# Last two args are unused but here for API compatibility
def get_cached_qasm(hs_filename, prog_args=None, mute=False, use_sudo=False):
    hs_basename = os.path.basename(hs_filename)
    basename = '-'.join([hs_basename] + list(prog_args or [])) + '.qasm'
    qasm_path = os.path.join(QUIPPER_QASM_DIR, basename)
    with open(qasm_path) as fp:
        return fp.read()

def get_qasm_str(*args, **kwargs):
    f = get_cached_qasm if QUIPPER_QASM_DIR else run_docker
    return f(*args, **kwargs)

def main(args):
    if len(args) < 2:
        print(f'usage: {args[0]} <filename>', file=sys.stderr)
        return 1
    filename = args[1]
    prog_args = args[2:]

    qasm_str = get_qasm_str(filename, prog_args)
    circ = qiskit.qasm3.loads(qasm_str)
    #print(circ.draw('text'))
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
