"""
written by Claude to fix Windows DLL loading issues in conda environment.
CRITICAL: Windows DLL Fix - Must be FIRST before ANY other imports!
"""

import os
import sys
import ctypes

if sys.platform == 'win32':
    # Get conda environment path (sys.executable = C:\...\envs\daniel\python.exe)
    env_path = os.path.dirname(sys.executable)
    dll_dir = os.path.join(env_path, 'Library', 'bin')

    if os.path.exists(dll_dir):
        # Add DLL directory to PATH (must be first!)
        os.environ['PATH'] = dll_dir + os.pathsep + os.environ.get('PATH', '')
        os.add_dll_directory(dll_dir)

        # Preload critical MKL DLLs to avoid conflicts
        for dll_name in ['mkl_core.2.dll', 'mkl_intel_thread.2.dll', 'mkl_def.2.dll']:
            dll_path = os.path.join(dll_dir, dll_name)
            if os.path.exists(dll_path):
                try:
                    ctypes.CDLL(dll_path)
                except:
                    pass

    # Disable Qt to avoid GUI conflicts
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''
    
    # PyCharm-specific: Disable automatic Qt backend selection
    # This prevents PyCharm's IPython from trying to load Qt
    os.environ['QT_API'] = ''
    os.environ['_PYDEV_BUNDLE_'] = 'pydevd'  # Tell PyCharm to not use Qt
    
    # Force matplotlib to use Agg backend (non-interactive, debugging-safe)
    # or use 'TkAgg' if you need interactive plots
    os.environ['MPLBACKEND'] = 'TkAgg'

