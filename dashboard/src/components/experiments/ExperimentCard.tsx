import React from 'react'
import { Link } from 'react-router-dom'
import { ExperimentSummary } from '@types/experiment'

interface ExperimentCardProps {
  experiment: ExperimentSummary
}

export const ExperimentCard: React.FC<ExperimentCardProps> = ({ experiment }) => {
  const statusColors = {
    running: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    completed: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200',
    failed: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
  }

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr)
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  const getDuration = (start: string, end: string) => {
    const startDate = new Date(start)
    const endDate = new Date(end)
    const diffMs = endDate.getTime() - startDate.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    const hours = Math.floor(diffMins / 60)
    const mins = diffMins % 60
    return hours > 0 ? `${hours}h ${mins}m` : `${mins}m`
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-6 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            {experiment.experiment_id}
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            Started {formatDate(experiment.start_time)}
          </p>
        </div>
        <span
          className={`px-3 py-1 rounded-full text-xs font-medium ${
            statusColors[experiment.status as keyof typeof statusColors] || statusColors.completed
          }`}
        >
          {experiment.status}
        </span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400">Rounds</p>
          <p className="text-sm font-medium text-gray-900 dark:text-white">
            {experiment.total_rounds}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400">Games</p>
          <p className="text-sm font-medium text-gray-900 dark:text-white">
            {experiment.total_games}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400">API Calls</p>
          <p className="text-sm font-medium text-gray-900 dark:text-white">
            {experiment.total_api_calls}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400">Cost</p>
          <p className="text-sm font-medium text-gray-900 dark:text-white">
            ${experiment.total_cost.toFixed(2)}
          </p>
        </div>
      </div>

      {experiment.end_time && (
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
          Duration: {getDuration(experiment.start_time, experiment.end_time)}
        </p>
      )}

      <div className="flex items-center justify-between">
        <div className="flex gap-2">
          <Link
            to={`/experiments/${experiment.experiment_id}`}
            className="px-4 py-2 text-sm font-medium text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 rounded-md"
          >
            View Details
          </Link>
          <Link
            to={`/experiments/${experiment.experiment_id}/visualizations`}
            className="px-4 py-2 text-sm font-medium text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded-md"
          >
            Visualizations
          </Link>
          <button
            className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 rounded-md"
            onClick={() => {
              // TODO: Implement export functionality
              console.log('Export', experiment.experiment_id)
            }}
          >
            Export
          </button>
        </div>
        <button
          className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 rounded-md"
          onClick={() => {
            // TODO: Implement share functionality
            console.log('Share', experiment.experiment_id)
          }}
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m9.032 4.026a3 3 0 10-5.432 3.536 5.065 5.065 0 011.432-3.536zm0-8.684a3 3 0 10-5.432-3.536 5.065 5.065 0 001.432 3.536z" />
          </svg>
        </button>
      </div>
    </div>
  )
}