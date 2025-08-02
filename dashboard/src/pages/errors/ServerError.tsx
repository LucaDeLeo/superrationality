import React from 'react'
import { Link } from 'react-router-dom'

export const ServerError: React.FC = () => {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
      <div className="max-w-md w-full text-center">
        <div className="mb-8">
          <h1 className="text-9xl font-bold text-gray-200 dark:text-gray-700">500</h1>
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mt-4">
            Server error
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Something went wrong on our servers. Please try again later.
          </p>
        </div>
        
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Link
            to="/"
            className="px-6 py-3 bg-primary-600 text-white rounded-md hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 transition-colors"
          >
            Go back home
          </Link>
          <button
            onClick={() => window.location.reload()}
            className="px-6 py-3 bg-gray-200 dark:bg-gray-700 text-gray-900 dark:text-white rounded-md hover:bg-gray-300 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 transition-colors"
          >
            Try again
          </button>
        </div>
      </div>
    </div>
  )
}