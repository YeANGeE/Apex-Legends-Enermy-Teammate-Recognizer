"""User-run recovery entry for an interrupted controller session."""
from c2_hidhide_session import restore

if __name__ == '__main__':
    restore()
    print('真实手柄会话设置已恢复；如游戏缓存设备状态，请重新连接手柄。')
