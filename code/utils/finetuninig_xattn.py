"""
Compatibility wrapper for a common typo: finetuninig_xattn.py

This delegates to run_finetuning_xattn.py.
"""

from run_finetuning_xattn import main


if __name__ == "__main__":
    main()
