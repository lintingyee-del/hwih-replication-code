# -*- coding: utf-8 -*-
"""Baidu QR-code login: saves the login QR as a PNG for the user to scan with
the mobile Baidu app, polls until confirmed, then writes the index.baidu.com
cookie string to data/baidu_cookie.txt and validates it.

No browser or DevTools needed. QR refreshes automatically if it expires.
"""

# Replication-package paths
from pathlib import Path as _ReplicationPath
import os as _ReplicationOS
_REP_PROJECT = _ReplicationPath(__file__).resolve().parents[1]
_REP_PACKAGE = _REP_PROJECT.parent
_REP_RESTRICTED = _REP_PACKAGE / 'restricted_data'
_REP_JUDGMENTS = _ReplicationPath(_ReplicationOS.environ.get(
    'HWIH_JUDGMENT_ARCHIVE', _REP_RESTRICTED / 'judgment_archive'))
_REP_CASE_ARCHIVE = _ReplicationPath(_ReplicationOS.environ.get(
    'HWIH_CASE_LEVEL_ARCHIVE', _REP_RESTRICTED / 'case_level_archive.parquet'))
_REP_MORTALITY = _ReplicationPath(_ReplicationOS.environ.get(
    'HWIH_CDC_SOURCE_ROOT', _REP_RESTRICTED / 'mortality_volumes'))
_REP_REGISTRY = _ReplicationPath(_ReplicationOS.environ.get(
    'HWIH_REGISTRY_ROOT', _REP_RESTRICTED / 'firm_registry'))
_REP_BAIDU = _ReplicationPath(_ReplicationOS.environ.get(
    'HWIH_BAIDU_ROOT', _REP_RESTRICTED / 'baidu'))
_REP_INTERVIEWS = _ReplicationPath(_ReplicationOS.environ.get(
    'HWIH_INTERVIEWS_ROOT', _REP_RESTRICTED / 'interviews'))
import os, sys, time

sys.stdout.reconfigure(encoding="utf-8")
from qdata.baidu_login.qr_login import (get_qrcode_info, get_bduss,
                                        get_login_cookie, get_exin, session)
from qdata.baidu_index import test_cookies

DATA = str(_REP_BAIDU)
QR_PATH = os.path.join(DATA, "baidu_qr.png")
COOKIE_FILE = os.path.join(DATA, "baidu_cookie.txt")

TOTAL_BUDGET = 600     # give up after 10 minutes
QR_LIFETIME = 170      # regenerate QR before Baidu expires it (~3 min)


def fetch_qr():
    link, sign, callback = get_qrcode_info()
    img = session.get(link, timeout=20).content
    with open(QR_PATH, "wb") as f:
        f.write(img)
    print(f"[QR] 二维码已保存: {QR_PATH}  (用手机百度App扫一扫并确认登录)", flush=True)
    return sign, callback


def main():
    os.makedirs(os.path.dirname(QR_PATH), exist_ok=True)
    t0 = time.time()
    sign, callback = fetch_qr()
    qr_born = time.time()
    bduss = None
    while time.time() - t0 < TOTAL_BUDGET:
        if time.time() - qr_born > QR_LIFETIME:
            print("[QR] 二维码即将过期,刷新一张,请重新打开图片扫码", flush=True)
            sign, callback = fetch_qr()
            qr_born = time.time()
        try:
            bduss = get_bduss(sign, callback)
            if bduss:
                break
        except Exception:
            pass          # still waiting for scan/confirm
        time.sleep(3)
    if not bduss:
        sys.exit("[FAIL] 10 分钟内未完成扫码,退出。重跑本脚本即可。")
    print("[OK] 扫码确认成功,换取登录 cookie ...", flush=True)
    cookies = get_login_cookie(bduss)
    try:
        cookies = cookies + get_exin()
    except Exception as e:
        print(f"[warn] 反爬附加 cookie 获取失败({type(e).__name__}),先用基础 cookie", flush=True)
    with open(COOKIE_FILE, "w", encoding="utf-8") as f:
        f.write(cookies)
    print(f"[OK] cookie 已写入 {COOKIE_FILE}", flush=True)
    ok = test_cookies(cookies)
    print(f"[VALIDATE] test_cookies => {'有效' if ok else '无效'}", flush=True)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
