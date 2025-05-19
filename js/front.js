
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({ 
    name: "hjh.facefusion.nodes",
    async setup() {
        const resp = await api.fetchApi("/get_facefusion_combo_set");
        app.FACEFUSION_COMBO_SET = await resp.json();
    },
    // async nodeCreated(node) {
    // },
    loadedGraphNode(node){
        // 设置不同换脸模型可用参数
        if(node.comfyClass == "FacefusionFaceSwapperProcessor"){
            node.widgets[0].callback=(value, widget, node)=>{
                const vals = app.FACEFUSION_COMBO_SET.face_swapper_models[value];
                node.widgets[1].options.values = vals;
                node.widgets[1].value = vals[0];
            }
        }

        if(node.comfyClass == "FacefusionFaceEnhancerProcessor"){
            const model_weight = get_widget_by_name(node.widgets,"model");
            if (model_weight.value != "codeformer") {
                set_disabled(node.widgets,"face_enhancer_weight",true)
            }
            model_weight.callback=(value, widget, node)=>{
                if (value == "codeformer") {
                    set_disabled(node.widgets,"face_enhancer_weight",false)
                }else{
                    set_disabled(node.widgets,"face_enhancer_weight",true)
                }
            }
        }

        if(node.comfyClass == "FacefusionOptionsNode"){
            widget_hidden(node.widgets);
            for(let i=0; i<node.widgets.length; i++){
                const widget = node.widgets[i];

                if(widget.name == "face_selector_mode"){
                    widget.callback = (value, widget, node) => {
                        set_disabled(node.widgets, "reference_face_index", value!="reference");
                    }
                    set_disabled(node.widgets, "reference_face_index", widget.value!="reference");
                }

                // 设置 face_mask 参数的显示隐藏
                if(widget.name == "use_face_box_mask"){
                    widget.callback = (value, widget, node) => {
                        set_disabled(node.widgets, "face_mask_", !value);
                    }
                    set_disabled(node.widgets, "face_mask_", !widget.value);
                }

                // 设置 face_region 参数的显示隐藏
                if(widget.name == "use_face_region_mask"){
                    widget.callback = (value, widget, node) => {
                        set_disabled(node.widgets, "face_region_", !value);
                    }
                    set_disabled(node.widgets, "face_region_", !widget.value);
                }

                // 设置 face_detector_size可用值列表
                if(widget.name == "face_detector_model"){
                    widget.callback=(value, widget, node)=>{
                        const _w = get_widget_by_name(node.widgets, "face_detector_size")
                        if(_w){
                            const vals = app.FACEFUSION_COMBO_SET.face_detector_set[value];
                            _w.options.values = vals;
                            _w.value = vals[vals.length-1];
                        }
                    }
                }
            }
        }

        if(node.comfyClass == "FacefusionProcesserOptionsNode"){
            widget_hidden(node.widgets);
        }
    },
});

function set_disabled(widgets,item_name_start, disabled){
    for(let i=0; i<widgets.length; i++){
        if(widgets[i].name.startsWith(item_name_start)){
            // widgets[i].hidden = !display;  
            widgets[i].disabled = disabled;
        }
    }
}
function widget_hidden(widgets){
    for(let i=0; i<widgets.length; i++){
        if(widgets[i].name.startsWith("hr_")){
            widgets[i].hidden =  true;
            widgets[i].disabled = true;
        }
    }
}
function get_widget_by_name(widgets, name){
    for(let i=0; i<widgets.length; i++){    
        if(widgets[i].name == name){
            return widgets[i];
        }
    }
    return null;
}