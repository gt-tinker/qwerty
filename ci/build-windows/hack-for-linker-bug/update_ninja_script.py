import os
import sys

def main():
    just_saw_exec_line = False

    with open('build.ninja', 'r') as fp, open('tmp.ninja', 'w') as nfp:
        for line in fp:
            if '.exe: CXX_EXECUTABLE_LINKER' in line:
                line = line.split('||')[0]
                # We may have lost our newline in the split above
                if not line.endswith('\n'):
                    line += '\n'
                line = line.replace('|', '')
                just_saw_exec_line = True
            elif just_saw_exec_line and 'LINK_LIBRARIES = ' in line:
                # It turns out there are still LINK_LIBRARIES that are not
                # dependencies. link.exe seems to be able to cope with them
                # because there's not many
                only_non_deps = ' '.join(arg for arg in line.split('=')[1].strip().split() if 'LLVM' not in arg and 'qwerty' not in arg)
                line = '  LINK_LIBRARIES = ' + only_non_deps + '\n'
                just_saw_exec_line = False

            nfp.write(line)

    os.replace('tmp.ninja', 'build.ninja')

    return 0

if __name__ == '__main__':
    sys.exit(main())
