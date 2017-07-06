from com.dtmilano.android.viewclient import ViewClient, View, ViewClientOptions

import types
import xml.etree.cElementTree as ET
import sys

HELP = 'help'
VERBOSE = 'verbose'
VERSION = 'version'
DEBUG = 'debug'
IGNORE_SECURE_DEVICE = 'ignore-secure-device'
IGNORE_VERSION_CHECK = 'ignore-version-check'
FORCE_VIEW_SERVER_USE = 'force-view-server-use'
DO_NOT_START_VIEW_SERVER = 'do-not-start-view-server'
DO_NOT_IGNORE_UIAUTOMATOR_KILLED = 'do-not-ignore-uiautomator-killed'
WINDOW = 'window'
ALL = 'all'
UNIQUE_ID = 'uniqueId'
POSITION = 'position'
BOUNDS = 'bounds'
CONTENT_DESCRIPTION = 'content-description'
TAG = 'tag'
CENTER = 'center'
SAVE_SCREENSHOT = 'save-screenshot'
SAVE_VIEW_SCREENSHOTS = 'save-view-screenshots'
DO_NOT_DUMP_VIEWS = 'do-not-dump-views'
DEVICE_ART = 'device-art'
DROP_SHADOW = 'drop-shadow'
SCREEN_GLARE = 'glare'
USE_UIAUTOMATOR_HELPER = 'use-uiautomator-helper'

MAP = {
    'a': View.__str__, ALL: View.__str__,
    'i': ViewClient.TRAVERSE_CITUI, UNIQUE_ID: ViewClient.TRAVERSE_CITUI,
    'x': ViewClient.TRAVERSE_CITPS, POSITION: ViewClient.TRAVERSE_CITPS,
    'b': ViewClient.TRAVERSE_CITB, BOUNDS: ViewClient.TRAVERSE_CITB,
    'd': ViewClient.TRAVERSE_CITCD, CONTENT_DESCRIPTION: ViewClient.TRAVERSE_CITCD,
    'g': ViewClient.TRAVERSE_CITG, TAG: ViewClient.TRAVERSE_CITG,
    'c': ViewClient.TRAVERSE_CITC, CENTER: ViewClient.TRAVERSE_CITC,
    'W': ViewClient.TRAVERSE_CITCDS, SAVE_VIEW_SCREENSHOTS: ViewClient.TRAVERSE_CITCDS,
    'D': ViewClient.TRAVERSE_S, DO_NOT_DUMP_VIEWS: ViewClient.TRAVERSE_S
}


class ViewClientHandler:
    @staticmethod
    def traverse(vc, root="ROOT", indent="", transform=None, stream=sys.stdout, pkg=''):
        '''
        Traverses the C{View} tree and prints its nodes.

        The nodes are printed converting them to string but other transformations can be specified
        by providing a method name as the C{transform} parameter.

        @type root: L{View}
        @param root: the root node from where the traverse starts
        @type indent: str
        @param indent: the indentation string to use to print the nodes
        @type transform: method
        @param transform: a method to use to transform the node before is printed
        '''

        if transform is None:
            # this cannot be a default value, otherwise
            # TypeError: 'staticmethod' object is not callable
            # is raised
            transform = ViewClient.TRAVERSE_CIT

        if type(root) == types.StringType and root == "ROOT":
            root = vc.root

        print vc.list()
        xml_root = ET.Element('hierarchy')
        ViewClientHandler.__traverse(root, indent, transform, stream, xml_node=xml_root, pkg=pkg)
        tree = ET.ElementTree(xml_root)
        tree.write("filename.xml")

        #         if not root:
        #             return
        #
        #         s = transform(root)
        #         if s:
        #             print >>stream, "%s%s" % (indent, s)
        #
        #         for ch in root.children:
        #             self.traverse(ch, indent=indent+"   ", transform=transform, stream=stream)

    @staticmethod
    def __traverse(root, indent="", transform=View.__str__, stream=sys.stdout, xml_node=None, pkg=''):
        if not root:
            return

        s = transform(root)
        sub_node = None
        if stream and s:
            ius = "%s%s" % (indent, s if isinstance(s, unicode) else unicode(s, 'utf-8', 'replace'))
            print >> stream, ius.encode('utf-8', 'replace')
            # print root.map
            if 'text:mText' in root.map:
                text = root.map['text:mText']
            else:
                text = ''
            if 'class' in root.map:
                node_class = root.map['class']
            else:
                node_class = ''
            if 'accessibility:getContentDescription()' in root.map:
                content_desc = root.map['accessibility:getContentDescription()']
                if content_desc == 'null':
                    content_desc = ''
            else:
                content_desc = ''
            bounds = str(root.getBounds()).replace('((', '[')
            bounds = bounds.replace('))', ']')
            bounds = bounds.replace('), (', '][')
            bounds = bounds.replace(', ', ',')

            print root.getPositionAndSize(), bounds
            if xml_node != None:
                sub_node = ET.SubElement(xml_node, "node", text=text, className=node_class, package=pkg,
                                         content_desc=content_desc, )
            else:
                xml_node = ET.Element('node', text=text)

        if sub_node == None:
            sub_node = xml_node

        for ch in root.children:
            ViewClientHandler.__traverse(ch, indent=indent + "   ", transform=transform, stream=stream, xml_node=sub_node,
                                   pkg=pkg)
        return sub_node

    @staticmethod
    def dump_view_server(package):
        kwargs1 = {VERBOSE: False, 'ignoresecuredevice': False, 'ignoreversioncheck': False}
        kwargs2 = {ViewClientOptions.FORCE_VIEW_SERVER_USE: False, ViewClientOptions.START_VIEW_SERVER: True,
                   ViewClientOptions.AUTO_DUMP: False, ViewClientOptions.IGNORE_UIAUTOMATOR_KILLED: True,
                   ViewClientOptions.COMPRESSED_DUMP: True,
                   ViewClientOptions.USE_UIAUTOMATOR_HELPER: False,
                   ViewClientOptions.DEBUG: {},
                   }
        kwargs2[ViewClientOptions.FORCE_VIEW_SERVER_USE] = True
        vc = ViewClient(*ViewClient.connectToDeviceOrExit(**kwargs1), **kwargs2)
        options = {WINDOW: -1, SAVE_SCREENSHOT: None, SAVE_VIEW_SCREENSHOTS: None, DO_NOT_DUMP_VIEWS: False,
                   DEVICE_ART: None, DROP_SHADOW: False, SCREEN_GLARE: False}
        windows = vc.list()
        print windows
        transform = MAP['b']
        for window in windows:
            if package not in windows[window]:
                continue
            print windows[window]
            vc.dump(window=int(window))
            ViewClient.imageDirectory = options[SAVE_VIEW_SCREENSHOTS]
            ViewClientHandler.traverse(vc, transform=transform, pkg=windows[window])



