import sys
import ast

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please specify pysl to validate')

    try:
        with open(sys.argv[1], 'r') as file:
            text = file.read()
        ast.parse(text)
        print('Validated syntax for: {0}'.format(sys.argv[1]))
    except IOError as e:
        print('Failed to open: {0} with error{1}'.format(sys.argv[1], e))
    except SyntaxError as e:
        print('Syntax error: {0}'.format(e))
