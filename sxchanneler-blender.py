bl_info = {
    'name': 'SX Channeler',
    'author': 'Jani Kahrama / Secret Exit Ltd.',
    'version': (0, 0, 1),
    'blender': (3, 5, 0),
    'location': 'View3D',
    'description': 'Channel Copy Tool',
    'doc_url': '',
    'tracker_url': 'https://github.com/FrandSX/sxchanneler-blender/issues',
    'category': 'Development',
}

import bpy


# ------------------------------------------------------------------------
#    Globals
# ------------------------------------------------------------------------
class SXCHANNELER_sxc_globals(object):
    def __init__(self):
        self.selection_modal_status = False
        self.prev_selection = []
        self.copy_buffer = {}

    def __del__(self):
        print('SX Channeler: Exiting sxc_globals')


# ------------------------------------------------------------------------
#    Color Conversions
# ------------------------------------------------------------------------
class SXCHANNELER_convert(object):
    def __init__(self):
        return None


    def color_to_luminance(self, in_rgba, premul=True):
        lumR = 0.212655
        lumG = 0.715158
        lumB = 0.072187
        alpha = in_rgba[3]

        linLum = lumR * in_rgba[0] + lumG * in_rgba[1] + lumB * in_rgba[2]
        # luminance = convert.linear_to_srgb((linLum, linLum, linLum, alpha))[0]
        if premul:
            return linLum * alpha  # luminance * alpha
        else:
            return linLum


    def luminance_to_color(self, value):
        return (value, value, value, 1.0)


    def luminance_to_alpha(self, value):
        return (1.0, 1.0, 1.0, value)


    def srgb_to_linear(self, in_rgba):
        out_rgba = []
        for i in range(3):
            if in_rgba[i] < 0.0:
                out_rgba.append(0.0)
            elif 0.0 <= in_rgba[i] <= 0.0404482362771082:
                out_rgba.append(float(in_rgba[i]) / 12.92)
            elif 0.0404482362771082 < in_rgba[i] <= 1.0:
                out_rgba.append(((in_rgba[i] + 0.055) / 1.055) ** 2.4)
            elif in_rgba[i] > 1.0:
                out_rgba.append(1.0)
        out_rgba.append(in_rgba[3])
        return out_rgba


    def linear_to_srgb(self, in_rgba):
        out_rgba = []
        for i in range(3):
            if in_rgba[i] < 0.0:
                out_rgba.append(0.0)
            elif 0.0 <= in_rgba[i] <= 0.00313066844250063:
                out_rgba.append(float(in_rgba[i]) * 12.92)
            elif 0.00313066844250063 < in_rgba[i] <= 1.0:
                out_rgba.append(1.055 * in_rgba[i] ** (float(1.0)/2.4) - 0.055)
            elif in_rgba[i] > 1.0:
                out_rgba.append(1.0)
        out_rgba.append(in_rgba[3])
        return out_rgba


    def colors_to_values(self, colors, as_rgba=False):
        count = len(colors) // 4
        if as_rgba:
            for i in range(count):
                color = colors[(0+i*4):(4+i*4)]
                lum = self.color_to_luminance(color, premul=False)
                colors[(0+i*4):(4+i*4)] = [lum, lum, lum, color[3]]
            return colors
        else:
            values = [None] * count
            for i in range(count):
                values[i] = self.color_to_luminance(colors[(0+i*4):(4+i*4)])
            return values


    def values_to_colors(self, values, invert=False, as_alpha=False, with_mask=False, as_tuple=False):
        if invert:
            values = self.invert_values(values)

        count = len(values) 
        colors = [None] * count * 4
        for i in range(count):
            if as_alpha:
                colors[(0+i*4):(4+i*4)] = [1.0, 1.0, 1.0, values[i]]
            elif with_mask:
                alpha = 1.0 if values[i] > 0.0 else 0.0
                colors[(0+i*4):(4+i*4)] = [values[i], values[i], values[i], alpha]
            else:
                colors[(0+i*4):(4+i*4)] = self.luminance_to_color(values[i])

        if as_tuple:
            rgba = [None] * count
            for i in range(count):
                rgba[i] = tuple(colors[(0+i*4):(4+i*4)])
            return rgba
        else:
            return colors


    def invert_values(self, values):
        return [1.0 - value for value in values]


    def __del__(self):
        print('SX Channeler: Exiting convert')


# ------------------------------------------------------------------------
#    Value Generators and Utils
#    NOTE: Switching between EDIT and OBJECT modes is slow.
#          Make sure OBJECT mode is enabled before calling
#          any functions in this class!
# ------------------------------------------------------------------------
class SXCHANNELER_generate(object):
    def __init__(self):
        return None


    def color_list(self, obj, color, masklayer=None, as_tuple=False):
        count = len(obj.data.loops)
        colors = [color[0], color[1], color[2], color[3]] * count
        return self.mask_list(obj, colors, masklayer, as_tuple=as_tuple)


    def vertex_id_list(self, obj):
        ids = [None] * len(obj.data.vertices)
        obj.data.vertices.foreach_get('index', ids)
        return ids


    def empty_list(self, obj, channelcount):
        return [0.0] * len(obj.data.loops) * channelcount


    def __del__(self):
        print('SX Channeler: Exiting generate')


# ------------------------------------------------------------------------
#    Layer Functions
#    NOTE: Objects must be in OBJECT mode before calling layer functions,
#          use utils.mode_manager() before calling layer functions
#          to set and track correct state
# ------------------------------------------------------------------------
class SXCHANNELER_layers(object):
    def __init__(self):
        return None

    def copy_channel(self, obj, sourcelayerindex, sourcechannel, targetlayerindex, targetchannel):
        pass


    # wrapper for low-level functions, always returns layerdata in RGBA
    def get_layer(self, obj, sourcelayer, as_tuple=False, single_as_alpha=False, apply_layer_opacity=False):
        rgba_targets = ['COLOR', 'SSS', 'EMI', 'CMP']
        alpha_targets = {'OCC': 0, 'MET': 1, 'RGH': 2, 'TRN': 3}
        dv = [1.0, 1.0, 1.0, 1.0]

        if sourcelayer.layer_type in rgba_targets:
            values = self.get_colors(obj, sourcelayer.color_attribute)

        elif sourcelayer.layer_type in alpha_targets:
            source_values = self.get_colors(obj, sourcelayer.color_attribute)
            values = [None] * len(source_values)
            for i in range(len(source_values)//4):
                value_slice = source_values[(0+i*4):(4+i*4)]
                value = value_slice[alpha_targets[sourcelayer.layer_type]]
                if single_as_alpha:
                    if value > 0.0:
                        values[(0+i*4):(4+i*4)] = [dv[0], dv[1], dv[2], value]
                    else:
                        values[(0+i*4):(4+i*4)] = [0.0, 0.0, 0.0, value]
                else:
                    values[(0+i*4):(4+i*4)] = [value, value, value, 1.0]

        if apply_layer_opacity and sourcelayer.opacity != 1.0:
            count = len(values)//4
            for i in range(count):
                values[3+i*4] *= sourcelayer.opacity

        if as_tuple:
            count = len(values)//4
            rgba = [None] * count
            for i in range(count):
                rgba[i] = tuple(values[(0+i*4):(4+i*4)])
            return rgba

        else:
            return values


    # takes RGBA buffers, converts and writes to appropriate channels
    def set_layer(self, obj, colors, targetlayer):
        rgba_targets = ['COLOR', 'SSS', 'EMI', 'CMP']
        alpha_targets = {'OCC': 0, 'MET': 1, 'RGH': 2, 'TRN': 3}
        target_type = targetlayer.layer_type

        if target_type in rgba_targets:
            layers.set_colors(obj, targetlayer.color_attribute, colors)

        elif target_type in alpha_targets:
            target_values = self.get_colors(obj, 'Alpha Materials')
            values = layers.get_luminances(obj, sourcelayer=None, colors=colors, as_rgba=False)
            for i in range(len(values)):
                target_values[alpha_targets[target_type]+i*4] = values[i]
            self.set_colors(obj, 'Alpha Materials', target_values)


    def get_colors(self, obj, source_name):
        source_colors = obj.data.color_attributes[source_name].data
        colors = [None] * len(source_colors) * 4
        source_colors.foreach_get('color', colors)
        return colors


    def set_colors(self, obj, target, colors):
        target_colors = obj.data.color_attributes[target].data
        target_colors.foreach_set('color', colors)
        obj.data.update()


    def set_alphas(self, obj, target, values):
        colors = self.get_colors(obj, target)
        count = len(values)
        for i in range(count):
            color = colors[(0+i*4):(4+i*4)]
            color = [color[0], color[1], color[2], values[i]]
            colors[(0+i*4):(4+i*4)] = color
        target_colors = obj.data.color_attributes[target].data
        target_colors.foreach_set('color', colors)
        obj.data.update()  


    def get_luminances(self, obj, sourcelayer=None, colors=None, as_rgba=False, as_alpha=False):
        if colors is None:
            if sourcelayer is not None:
                colors = self.get_layer(obj, sourcelayer)
            else:
                colors = generate.empty_list(obj, 4)

        if as_rgba:
            values = generate.empty_list(obj, 4)
            count = len(values)//4
            for i in range(count):
                values[(0+i*4):(4+i*4)] = convert.luminance_to_color(convert.color_to_luminance(colors[(0+i*4):(4+i*4)]))
        elif as_alpha:
            values = generate.empty_list(obj, 4)
            count = len(values)//4
            for i in range(count):
                values[(0+i*4):(4+i*4)] = convert.luminance_to_alpha(convert.color_to_luminance(colors[(0+i*4):(4+i*4)]))
        else:
            values = generate.empty_list(obj, 1)
            count = len(values)
            for i in range(count):
                values[i] = convert.color_to_luminance(colors[(0+i*4):(4+i*4)])

        return values


    def get_uvs(self, obj, source, channel=None):
        channels = {'U': 0, 'V': 1}
        source_uvs = obj.data.uv_layers[source].data
        count = len(source_uvs)
        source_values = [None] * count * 2
        source_uvs.foreach_get('uv', source_values)

        if channel is None:
            uvs = source_values
        else:
            uvs = [None] * count
            sc = channels[channel]
            for i in range(count):
                uvs[i] = source_values[sc+i*2]

        return uvs


    # when targetchannel is None, sourceuvs is expected to contain data for both U and V
    def set_uvs(self, obj, target, sourceuvs, targetchannel=None):
        channels = {'U': 0, 'V': 1}
        target_uvs = obj.data.uv_layers[target].data

        if targetchannel is None:
            target_uvs.foreach_set('uv', sourceuvs)
        else:
            target_values = self.get_uvs(obj, target)
            tc = channels[targetchannel]
            count = len(sourceuvs)
            for i in range(count):
                target_values[tc+i*2] = sourceuvs[i]
            target_uvs.foreach_set('uv', target_values)


    def __del__(self):
        print('SX Channeler: Exiting layers')


# ------------------------------------------------------------------------
#    Core Functions
# ------------------------------------------------------------------------
def start_sxchanneler_modal():
    if (not sxc_globals.selection_modal_status) and (len(bpy.data.objects) > 0):
        bpy.ops.sxchanneler.selectionmonitor('EXEC_DEFAULT')
        sxc_globals.selection_modal_status = True


# Return value eliminates duplicates
def mesh_selection_validator(self, context):
    mesh_objs = []
    for obj in context.view_layer.objects.selected:
        if obj is None:
            pass
        elif (obj.type == 'MESH') and (obj.hide_viewport is False):
            mesh_objs.append(obj)

    return list(set(mesh_objs))


def update_sxchanneler_layers(self, context):
    obj = context.view_layer.objects.active
    if obj and obj.type == 'MESH':
        obj.sxchanneler_layers.clear()
        for layer in obj.data.color_attributes:
            item = obj.sxchanneler_layers.add()
            item.name = layer.name
            item.type = 'COLOR'

        for layer in obj.data.uv_layers:
            item = obj.sxchanneler_layers.add()
            item.name = layer.name
            item.type = 'UV'


# ------------------------------------------------------------------------
#    Scene Properties
# ------------------------------------------------------------------------
class SXCHANNELER_sceneprops(bpy.types.PropertyGroup):

    active_object: bpy.props.PointerProperty(
        name='Active Object',
        type=bpy.types.Object,
        update=update_sxchanneler_layers)


class SXCHANNELER_objectprops(bpy.types.PropertyGroup):

    source_rgba: bpy.props.EnumProperty(
        name='Source Color Channel',
        description='Select source channel',
        items=[
            ('R', 'R', ''),
            ('G', 'G', ''),
            ('B', 'B', ''),
            ('A', 'A', ''),
            ('L', 'LUM', '')],
        default='R')

    source_uv: bpy.props.EnumProperty(
        name='Source UV Channel',
        description='Select source channel',
        items=[
            ('U', 'U', ''),
            ('V', 'V', '')],
        default='U')

    target_rgba: bpy.props.EnumProperty(
        name='Target Color Channel',
        description='Select target channel',
        items=[
            ('R', 'R', ''),
            ('G', 'G', ''),
            ('B', 'B', ''),
            ('A', 'A', '')],
        default='R')

    target_uv: bpy.props.EnumProperty(
        name='Target UV Channel',
        description='Select target channel',
        items=[
            ('U', 'U', ''),
            ('V', 'V', '')],
        default='U')

    source_layer_index: bpy.props.IntProperty(
        name='Source Layer',
        description='Selected source layer',
        default=0)

    target_layer_index: bpy.props.IntProperty(
        name='Target Layer',
        description='Selected target layer',
        default=0)

    layer_count: bpy.props.IntProperty(
        name='Layer Count',
        default=0,
        update=update_sxchanneler_layers)


# ------------------------------------------------------------------------
#    UI Elements
# ------------------------------------------------------------------------
class SXCHANNELER_layer_item(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty()
    type: bpy.props.StringProperty()  # "UV" or "COLOR"


class SXCHANNELER_UL_layer_list(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layout.label(text=item.name)


class SXCHANNELER_PT_main_panel(bpy.types.Panel):
    bl_label = "SX Channeler"
    bl_idname = "SXCHANNELER_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Copy Paste Channels'


    def draw(self, context):
        scene = context.scene.sxchanneler
        layout = self.layout
        obj = context.view_layer.objects.active
        if obj and obj.type == 'MESH':
            if obj.sxchanneler.layer_count > 0:
                row_lists = layout.row(align=True)
                col_source = row_lists.column()
                col_source.label(text='Source:')
                col_source.template_list('SXCHANNELER_UL_layer_list', '', obj, 'sxchanneler_layers', obj.sxchanneler, 'source_layer_index')
                col_target = row_lists.column()
                col_target.label(text='Target:')
                col_target.template_list('SXCHANNELER_UL_layer_list', '', obj, 'sxchanneler_layers', obj.sxchanneler, 'target_layer_index')

                layer_types = {'COLOR': 'source_rgba', 'UV': 'source_uv'}
                split_channels = layout.split()
                row_source = split_channels.row(align=True)
                row_source.prop(obj.sxchanneler, layer_types[obj.sxchanneler_layers[obj.sxchanneler.source_layer_index].type], expand=True)

                layer_types = {'COLOR': 'target_rgba', 'UV': 'target_uv'}
                row_target = split_channels.row(align=True)
                row_target.prop(obj.sxchanneler, layer_types[obj.sxchanneler_layers[obj.sxchanneler.target_layer_index].type], expand=True)

                row = layout.row()
                row.operator('sxchanneler.copy')
            else:
                layout.label(text='No data layers on selected object')

        else:
            layout.label(text='No mesh object selected')


# ------------------------------------------------------------------------
#   Operators
# ------------------------------------------------------------------------

# The selectionmonitor tracks shading mode and selection changes,
# triggering a refresh of palette swatches and other UI elements
class SXCHANNELER_OT_selectionmonitor(bpy.types.Operator):
    bl_idname = 'sxchanneler.selectionmonitor'
    bl_label = 'Selection Monitor'
    bl_description = 'Refreshes the UI on selection change'


    @classmethod
    def poll(cls, context):
        if not bpy.app.background:
            if (context.area is not None) and (context.area.type == 'VIEW_3D'):
                return context.active_object is not None
            else:
                return False
        else:
            return context.active_object is not None


    def modal(self, context, event):
        if event.type == 'TIMER_REPORT':
            return {'PASS_THROUGH'}

        if not context.area:
            print('Selection Monitor: Context Lost')
            sxc_globals.selection_modal_status = False
            return {'CANCELLED'}

        objs = mesh_selection_validator(self, context)
        if (len(objs) == 0) and (context.active_object is not None) and (context.object.mode == 'EDIT'):
            objs = context.objects_in_mode
            if objs:
                for obj in objs:
                    obj.select_set(True)
                context.view_layer.objects.active = objs[0]

        if objs:
            obj = context.view_layer.objects.active
            if obj != context.scene.sxchanneler.active_object:
                context.scene.sxchanneler.active_object = obj
            if len(obj.sxchanneler_layers) != obj.sxchanneler.layer_count:
                obj.sxchanneler.layer_count = len(obj.sxchanneler_layers)

                return {'PASS_THROUGH'}
            else:
                return {'PASS_THROUGH'}

        return {'PASS_THROUGH'}


    def execute(self, context):
        return self.invoke(context, None)


    def invoke(self, context, event):
        context.window_manager.modal_handler_add(self)
        print('SX Channeler: Starting selection monitor')
        return {'RUNNING_MODAL'}


class SXCHANNELER_OT_copy(bpy.types.Operator):
    bl_idname = "sxchanneler.copy"
    bl_label = "Copy Channel"

    def execute(self, context):
        obj = context.view_layer.objects.active
        sourcelayerindex = obj.sxchanneler.source_layer_index
        if obj.sxchanneler_layers[obj.sxchanneler.source_layer_index].type == 'COLOR':
            sourcechannel = obj.sxchanneler.source_rgba
        else:
            sourcechannel = obj.sxchanneler.source_uv

        targetlayerindex = obj.sxchanneler.target_layer_index
        if obj.sxchanneler_layers[obj.sxchanneler.target_layer_index].type == 'COLOR':
            targetchannel = obj.sxchanneler.target_rgba
        else:
            targetchannel = obj.sxchanneler.target_uv

        layers.copy_channel(obj, sourcelayerindex, sourcechannel, targetlayerindex, targetchannel)

        return {'FINISHED'}


# ------------------------------------------------------------------------
#    Registration and initialization
# ------------------------------------------------------------------------
sxc_globals = SXCHANNELER_sxc_globals()
convert = SXCHANNELER_convert()
generate = SXCHANNELER_generate()
layers = SXCHANNELER_layers()

classes = (
    SXCHANNELER_sceneprops,
    SXCHANNELER_objectprops,
    SXCHANNELER_layer_item,
    SXCHANNELER_UL_layer_list,
    SXCHANNELER_PT_main_panel,
    SXCHANNELER_OT_selectionmonitor,
    SXCHANNELER_OT_copy)


def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)

    bpy.types.Object.sxchanneler = bpy.props.PointerProperty(type=SXCHANNELER_objectprops)
    bpy.types.Scene.sxchanneler = bpy.props.PointerProperty(type=SXCHANNELER_sceneprops)
    bpy.types.Object.sxchanneler_layers = bpy.props.CollectionProperty(type=SXCHANNELER_layer_item)
    if not bpy.app.background:
        bpy.types.SpaceView3D.draw_handler_add(start_sxchanneler_modal, (), 'WINDOW', 'POST_VIEW')


def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)

    del bpy.types.Object.sxchanneler_layers
    del bpy.types.Scene.sxchanneler
    del bpy.types.Object.sxchanneler

    if not bpy.app.background:
        bpy.types.SpaceView3D.draw_handler_remove(start_sxchanneler_modal, 'WINDOW')


if __name__ == "__main__":
    try:
        unregister()
    except:
        pass
    register()
