import sys

def main():
    new_lines = []
    just_saw_exec_line = False

    with open('build.ninja', 'r') as fp:
        for line in fp:
            if '.exe: CXX_EXECUTABLE_LINKER' in line:
                line = line.split('||')[0]
                # We may have lost our newline in the split above
                if not line.endswith('\n'):
                    line += '\n'
                line = line.replace('|', '')
                just_saw_exec_line = True
                new_lines.append(line)
            elif just_saw_exec_line and 'LINK_LIBRARIES = ' in line:
                only_non_deps = ' '.join(arg for arg in line.split('=')[1].strip().split() if 'LLVM' not in arg and 'qwerty' not in arg)
                line = '  LINK_LIBRARIES = ' + only_non_deps + '\n'
                new_lines.append(line)
                just_saw_exec_line = False
            else:
                new_lines.append(line)

    with open('build.ninja', 'w') as fp:
        fp.writelines(new_lines)

    return 0

if __name__ == '__main__':
    sys.exit(main())
