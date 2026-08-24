import PyQt5, os
p = os.path.dirname(PyQt5.__file__)
print('PyQt5 path:', p)
for r, ds, fs in os.walk(p):
    for f in fs:
        if f.startswith('Qt5') or f == 'qwindows.dll' or f == 'qt.conf':
            print(os.path.join(r, f))
