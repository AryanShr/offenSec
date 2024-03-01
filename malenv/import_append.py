import lief

def check_imports(PEFile):
    for imported_library in PEFile.imports:
      print(f"Library: {imported_library.name}")
      for function in imported_library.entries:
          print(f"  Function: {function.name}")

pe = lief.PE.parse("hello.exe")
builder = lief.PE.Builder(pe)
pe.add_library("kernel32.dll")
builder.build_imports()
# builder.patch_imports()

builder.build()
builder.write("hello_patched.exe")

pf = lief.PE.parse("hello_patched.exe")
check_imports(pf)