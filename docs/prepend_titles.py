import os
def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + '*'*len(line) + '\n\n' + content)

if __name__ == '__main__':
    for filename in os.listdir('api'):
        if filename.endswith('.rst'):
            # line_prepender('api/' + filename, '.. title:: ' + filename[:-4])
            line_prepender('api/' + filename, filename[:-4])