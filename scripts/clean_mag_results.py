#!/usr/bin/env python3
"""Cleanup helper для legacy magnetic_pump artifacts (ТЗ §4.5).

Сценарии:
  --move    → переместить legacy-файлы из flat-directory в `_legacy/`
              (по умолчанию)
  --delete  → удалить legacy-файлы
  --dry-run → только перечислить

Legacy: всё, что лежит непосредственно в results/magnetic_pump/ и не
является run-каталогом (run-каталог содержит manifest.json). Также
отдельно отмечаются известные legacy имена.
"""
import sys
import os
import shutil
import argparse


LEGACY_NAMES = {
    "mag_smooth_summary.json",
    "mag_textured_summary.json",
    "mag_smooth_equilibrium.csv",
    "mag_textured_equilibrium.csv",
    "pmax_vs_mag_share.png",
    "ratios_tex_vs_smooth.png",
    "eps_vs_unload_share.png",
    "hmin_vs_unload_share.png",
    "pmax_vs_unload_share.png",
    "force_sharing.png",
    "report.md",
}

# Не трогаем:
KEEP_FILES = {
    "latest_run.txt",  # pointer на текущий run_id
}


def is_run_dir(path):
    return (os.path.isdir(path)
            and os.path.exists(os.path.join(path, "manifest.json")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default=None,
                        help="magnetic_pump directory (default: "
                             "results/magnetic_pump)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--move", action="store_true", default=True,
                      help="move legacy files to _legacy/ (default)")
    mode.add_argument("--delete", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = args.base or os.path.join(
        os.path.dirname(__file__), "..", "results", "magnetic_pump")
    base = os.path.abspath(base)
    if not os.path.isdir(base):
        print(f"Нет {base}")
        return

    legacy_dir = os.path.join(base, "_legacy")
    legacy_items = []
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if name == "_legacy":
            continue
        if name in KEEP_FILES:
            continue
        if is_run_dir(path):
            # valid magnetic_v4 run — не трогать
            continue
        # Прочее: files in flat-dir или каталоги без manifest → legacy
        if os.path.isfile(path) or os.path.isdir(path):
            legacy_items.append(path)

    if not legacy_items:
        print("Legacy files не найдены.")
        return

    print(f"Legacy items ({len(legacy_items)}):")
    for p in legacy_items:
        rel = os.path.relpath(p, base)
        tag = " (known-legacy)" if os.path.basename(p) in LEGACY_NAMES else ""
        print(f"  {rel}{tag}")

    if args.dry_run:
        print("\n--dry-run: ничего не меняю.")
        return

    if args.delete:
        for p in legacy_items:
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
        print(f"\nУдалено: {len(legacy_items)} items")
        return

    # default: move
    os.makedirs(legacy_dir, exist_ok=True)
    for p in legacy_items:
        dst = os.path.join(legacy_dir, os.path.basename(p))
        if os.path.exists(dst):
            base_dst, ext = os.path.splitext(dst)
            k = 1
            while os.path.exists(f"{base_dst}.{k}{ext}"):
                k += 1
            dst = f"{base_dst}.{k}{ext}"
        shutil.move(p, dst)
    print(f"\nПеремещено в {legacy_dir}: {len(legacy_items)} items")


if __name__ == "__main__":
    main()
