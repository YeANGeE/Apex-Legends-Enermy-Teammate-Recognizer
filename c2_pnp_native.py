"""Read-only current-device inventory via cfgmgr32; no WMI or PowerShell."""
import ctypes as c
import re


class InventoryNotReady(RuntimeError):
    pass


def controller_rows(vid_pid='VID_3537&PID_10C6'):
    api=c.WinDLL('cfgmgr32')
    u=c.c_uint32; pu=c.POINTER(u); w=c.c_wchar_p
    signatures={
        'CM_Get_Device_ID_List_SizeW':[pu,w,u],
        'CM_Get_Device_ID_ListW':[w,c.POINTER(c.c_wchar),u,u],
        'CM_Locate_DevNodeW':[pu,w,u],
        'CM_Get_DevNode_Registry_PropertyW':[u,u,pu,c.c_void_p,pu,u],
        'CM_Get_DevNode_Status':[pu,pu,u,u],
        'CM_Get_Parent':[pu,u,u],
        'CM_Get_Device_IDW':[u,c.POINTER(c.c_wchar),u,u],
    }
    for name,args in signatures.items():
        function=getattr(api,name);function.argtypes=args;function.restype=u
    def check(result,operation):
        if result:
            raise InventoryNotReady(f'{operation} returned CONFIGRET 0x{result:08x}')
    for _ in range(3):
        size=u()
        check(api.CM_Get_Device_ID_List_SizeW(c.byref(size),None,0x100),'Device list size')
        if not 1<=size.value<=2_000_000:
            raise RuntimeError('Invalid native device list size')
        buffer=c.create_unicode_buffer(size.value)
        result=api.CM_Get_Device_ID_ListW(None,buffer,size.value,0x100)
        if result==0x1a:  # CR_BUFFER_SMALL: list changed between size and fetch.
            continue
        check(result,'Device list')
        ids=[path for path in buffer[:].split('\0') if vid_pid in path.upper()]
        break
    else:
        raise InventoryNotReady('Device list changed repeatedly')
    rows={}
    def read_row(path,node):
        path=path.upper()  # Instance IDs are case-insensitive; stable lease representation.
        name=c.create_unicode_buffer(256);length=u(c.sizeof(name));kind=u()
        check(api.CM_Get_DevNode_Registry_PropertyW(node,8,c.byref(kind),name,c.byref(length),0),'Device class')
        status=u();problem=u()
        check(api.CM_Get_DevNode_Status(c.byref(status),c.byref(problem),node,0),'Device status')
        root=re.fullmatch(r'USB\\'+re.escape(vid_pid)+r'\\[^\\]+',path,re.IGNORECASE)
        if name.value not in ('HIDClass','XnaComposite') and not (name.value=='USB' and root):
            return
        rows[path]=dict(Class=name.value,InstanceId=path,Status='OK' if problem.value==0 and status.value&8 else 'ERROR')
        if name.value=='XnaComposite':
            parent=u();check(api.CM_Get_Parent(c.byref(parent),node,0),'Device parent')
            parent_path=c.create_unicode_buffer(512)
            check(api.CM_Get_Device_IDW(parent.value,parent_path,len(parent_path),0),'Parent identity')
            if re.fullmatch(r'USB\\'+re.escape(vid_pid)+r'\\[^\\]+',parent_path.value,re.IGNORECASE):
                read_row(parent_path.value,parent.value)
    for path in ids:
        node=u();check(api.CM_Locate_DevNodeW(c.byref(node),path,0),'Locate current device')
        read_row(path,node.value)
    return [rows[path] for path in sorted(rows)]
